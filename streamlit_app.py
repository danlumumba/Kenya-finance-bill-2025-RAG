# -*- coding: utf-8 -*-
"""finance_bill_rag.ipynb
## RAG ##
Retrieval Augmented Generation
- Retrieval - Fetching data from a database or a large data source
- Augmented - To enhance. Brings together answers that have been retrieved by the retriever.
- Generation - Try to make sense of what it has been augmented and generate an output.

> - Vector databases - A vector database is a specialized type of database designed to store, index, and search vector embeddings—which are numerical representations of data like text, images, audio, or video.
"""

import streamlit as st
from langchain.retrievers import ParentDocumentRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

st.title("🇰🇪 Kenya Finance Bill 2025 - Q&A")

load_dotenv()

#groq_api_key = st.secrets["GROQ_API_KEY"]
load_dotenv()
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ API key not loaded successfully")

# api_key = os.getenv('GROQ_API_KEY')

# if not api_key:
#   raise ValueError('GROQ API key not loaded successful')
# else:
#   print('API key loaded successfully')

pdf_link = r'C:\Users\NGARE\DS_Python_Lux\Ds_projects\FinanceBillRag\The Finance Bill 2025.pdf'

# Loading the PDF file
loader = PyPDFLoader(pdf_link)
docs = loader.load()

print(len(docs))

parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 700)
parent_docs = parent_splitter.split_documents(docs)

print(len(parent_docs))
child_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 700)

#initializing the embedding model
bge_model = SentenceTransformer("BAAI/bge-base-en")

#creating the embedding class
class BGEEmbeddings:
  def embed_documents(self, text):
    """Generate Embeddings for batch of documents"""
    return bge_model.encode(text, batch_size = 8, normalize_embeddings = True).tolist()
  def embed_query(self, text):
    """Generate Embeddings for a single query"""
    return bge_model.encode([text],normalize_embeddings = True).tolist()[0]


from langchain.embeddings.base import Embeddings

class LangchainBGEEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return BGEEmbeddings().embed_documents(texts)

    def embed_query(self, text):
        return BGEEmbeddings().embed_query(text)
embedding = LangchainBGEEmbeddings()

# Setup Vector Store and Retriever

store = InMemoryStore()
vectorstore = Chroma(collection_name="kenya_finance_bill",
                     embedding_function=embedding,
                     persist_directory = 'finance_bill_vectorstore')

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter
)

#Add Documents to the Retriever

retriever.add_documents(parent_docs)
print("Documents indexed into retriever")

# Initialize the Groq LLM

llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-70b-8192",
    api_key=api_key
)

#adding memory to the RAG
# A temperature of 0.0 means the LLM will produce the most deterministic and consistent (least "creative") output
# The memory variable will store the conversational memory component.
# The conversation variable will hold the core RAG conversational chain
llm = ChatGroq(temperature = 0.0, model_name = "llama3-70b-8192", api_key = api_key)
memory = ConversationBufferMemory(
    memory_key = "chat_history",
    return_messages = True
)
conversation = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever = retriever,
    memory = memory,
    verbose = True
)

#(Optional) Custom Prompt Template

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Use the following context to answer the question.
If the answer is not in the context, say "I can only answer questions related to Kenya finance bill 2025."

Context:
{context}

Question:
{question}

Answer:"""
)

# #Create RetrievalQA Chain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

#Ask a Question

query = "What are the proposed tax changes in the Finance Bill 2025?"
response = qa_chain.run(query)
print("Answer:", response)

#Question 2

query = "What are the proposed income tax bands or rates?"
response = qa_chain.run(query)
print("Answer:", response)

#Question 3

query = "How does the bill affect small businesses or startups?"
response = qa_chain.run(query)
print("Answer:", response)

conversation("What are the proposed tax changes in the Finance Bill 2025?")

conversation("What are the proposed income tax bands or rates?")

conversation("")
