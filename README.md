# Kenya-finance-bill-2025-RAG
This is a simple Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about the Kenya Finance Bill 2025 and receive intelligent, context-aware answers.

# 🇰🇪 Kenya Finance Bill 2025 - RAG Q&A System

This is a simple Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about the **Kenya Finance Bill 2025** and receive intelligent, context-aware answers.

---

## 🚀 Features

- ✅ Loads and processes the full Finance Bill text.
- ✅ Splits the text into manageable chunks with context overlap.
- ✅ Uses vector embeddings for efficient similarity search.
- ✅ Employs a Large Language Model (LLM) to generate accurate responses.
- ✅ Supports conversational memory (optional).
- ✅ Fully implemented in Python using LangChain and FAISS.

---

## 🧠 How It Works

1. **Ingest Bill Text**  
   The full text is split into overlapping chunks to maintain context.

2. **Embed the Text**  
   Chunks are converted into numerical vectors using a transformer model.

3. **Store in Vector DB**  
   Embeddings are indexed using FAISS for fast similarity search.

4. **Query and Retrieve**  
   User queries are converted to vectors, and relevant chunks are retrieved.

5. **Answer with LLM**  
   A language model (like GPT-3.5) generates a response using the retrieved chunks.

---

## 🛠️ Technologies Used

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI GPT or Mistral](https://platform.openai.com/)
- [Python](https://www.python.org/)
- Optional: Streamlit or Gradio for UI

---

## 📦 Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/kenya-finance-bill-rag.git
   cd kenya-finance-bill-rag
