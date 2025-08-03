import os
import time
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Load environment variables from a .env file
load_dotenv()

# Apply the nest_asyncio patch for Streamlit compatibility
import nest_asyncio
nest_asyncio.apply()


# --- Helper Functions ---

def extract_text_from_pdf(file: BytesIO) -> list[str]:
    """Extracts text from a PDF file and splits it into chunks."""
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(full_text)

def setup_vectorstore(text_chunks: list[str], index_name: str, embeddings):
    """
    Sets up the Pinecone vector store.
    This function will clear any existing data in the index to ensure
    we are only querying the most recently uploaded PDF.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if index_name in pc.list_indexes().names():
        st.write(f"Clearing existing index '{index_name}'...")
        index = pc.Index(index_name)
        index.delete(delete_all=True)
    else:
        st.write(f"Creating new index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        time.sleep(2) # Wait for the index to be ready

    documents = [Document(page_content=chunk) for chunk in text_chunks]
    
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=index_name
    )
    st.write("Pinecone index is ready.")

def create_rag_chain(index_name: str, embeddings):
    """Creates the RAG chain for question answering."""
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    prompt_template = """
    You are an expert assistant for question-answering tasks.
    Use the following retrieved context and the chat history to answer the question.
    If you don't know the answer, just say that you don't know.
    Be concise and helpful.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --- Streamlit App ---

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("📄 PDF Q&A with Gemini & Pinecone")
st.markdown("Upload a PDF and ask questions about its content. The bot will use the document to answer.")

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    st.error("🚨 Missing API keys. Please add `GOOGLE_API_KEY` and `PINECONE_API_KEY` to your `.env` file.")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("Upload your PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if pdf_file:
        if st.button("Process Document"):
            with st.spinner("Reading, chunking, and indexing your document..."):
                chunks = extract_text_from_pdf(pdf_file)
                
                if not chunks:
                    st.warning("Could not extract any text from the PDF. It might be empty or contain only images.")
                    st.stop()
                
                index_name = "rag-pdf-qa" 
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                
                setup_vectorstore(chunks, index_name, embeddings)
                
                st.session_state.rag_chain = create_rag_chain(index_name, embeddings)
                
                st.session_state.chat_history = []
                st.success("Document processed! You can now ask questions.")

st.header("Chat with the Document")

for q, a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        st.markdown(a)

question = st.chat_input("Ask a question about the document...")

if question:
    if not st.session_state.rag_chain:
        st.warning("Please upload and process a document first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history_str = "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in st.session_state.chat_history])
                
                response = st.session_state.rag_chain.invoke({
                    "question": question,
                    "chat_history": history_str
                })
                st.markdown(response)
        
        st.session_state.chat_history.append((question, response))