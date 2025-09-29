# 📚 Retrieval-Augmented Generation (RAG) Chatbot

A domain-specific chatbot built with **LangChain**, **Hugging Face embeddings**, and **FAISS/Chroma** vector search.  
This project ingests PDFs (e.g., research papers, energy reports), creates embeddings, stores them in FAISS/Chroma, and lets you ask natural language questions.  

## 🚀 Features
- PDF ingestion & text chunking
- Hugging Face `sentence-transformers` embeddings
- Vector search with FAISS (local) or Chroma
- LLM integration (Hugging Face / OpenAI optional)
- Function-calling for external APIs (e.g., Wikipedia)
- Gradio UI for interactive chat

## 📂 Project Structure
will be added

## 🛠 Quickstart (in Colab)
Open `notebooks/colab_demo.ipynb` in Colab and run the cells.

## 🤝 Contributions
PRs welcome — this project is structured to be extended with more embeddings, LLMs, or vector stores.
