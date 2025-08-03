# 📄 DocuChat: Conversational AI-Powered Q&A over PDF Documents

**DocuChat** is a full-stack AI application that enables users to interact with PDF documents in a conversational manner. Powered by Google's Gemini Pro and Pinecone, it uses a Retrieval-Augmented Generation (RAG) architecture to generate accurate, context-aware answers to questions based on the uploaded PDF.

Built with **Streamlit**, it offers an intuitive web interface for real-time document understanding, making it ideal for use cases such as research papers, reports, eBooks, or business documents.

---

## 🚀 Live Demo

👉 [Launch the App](https://docuchat-fsj483xk5zefo9s5ccxdl6.streamlit.app/)

> *Hosted on Streamlit Cloud — no setup required.*

---

## 📚 Use Cases

- 📘 **Educational**: Ask questions from lecture notes, textbooks, or research papers.
- 🧾 **Corporate**: Explore annual reports, policies, or business documents.
- ⚖️ **Legal/Medical**: Query lengthy contracts, reports, or case files.
- 🤖 **Knowledge Assistants**: Build intelligent document chatbots.

---

## 🧠 Features

✅ Upload a PDF document  
✅ Extract and chunk document content  
✅ Store and retrieve using semantic vector search (Pinecone)  
✅ Generate contextual answers using Gemini Pro (LLM)  
✅ Maintain conversational memory  
✅ Clean, responsive frontend using Streamlit  
✅ Secure API key handling via `.env` or Streamlit Secrets

---

## 🛠️ Tech Stack

| Layer        | Technology                              | Description                                         |
|--------------|------------------------------------------|-----------------------------------------------------|
| Frontend     | **Streamlit**                           | Minimalistic web app UI                            |
| LLM          | **Google Gemini Pro (Chat + Embeddings)** | Question answering + semantic vector generation     |
| Vector Store | **Pinecone**                            | Fast similarity search for retrieved chunks         |
| Framework    | **LangChain**                           | RAG pipeline, memory, prompt management             |
| Parsing      | **PyPDF**                               | Extracts raw text from PDF                         |
| Hosting      | **Streamlit Cloud**                     | Free hosting for Python web apps                    |

---

## 🧾 PDF → Q&A Pipeline Overview

        ┌──────────────┐
        │  PDF Upload  │
        └─────┬────────┘
              │
              ▼
    ┌────────────────────┐
    │ Text Extraction    │ ← PyPDF
    └────────────────────┘
              │
              ▼
    ┌────────────────────┐
    │ Chunking           │ ← LangChain TextSplitter
    └────────────────────┘
              │
              ▼
    ┌────────────────────┐
    │ Embedding Chunks   │ ← Gemini Embeddings
    └────────────────────┘
              │
              ▼
    ┌────────────────────┐
    │ Pinecone Indexing  │ ← VectorStore
    └────────────────────┘
              │
              ▼
    ┌────────────────────┐
    │ User Question      │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────────────┐
    │ RAG + Prompt + Gemini Chat │ ← LangChain + Gemini
    └────────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ Response Displayed │ ← Streamlit
    └────────────────────┘

Made with 💻 by Yash Kahalkar, Email: kahalkaryash@gmail.com 
