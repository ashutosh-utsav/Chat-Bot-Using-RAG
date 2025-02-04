import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

os.environ["GOOGLE_API_KEY"] = api_key

# Load PDF before running chatbot
pdf_path = "/home/ashutosh/Codes/Projects/Rag_Based_Chat_Bot/Ashutosh_Resume.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split Document into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# Convert Text to Embeddings (FIXED MODEL NAME)
embedding_model = GoogleGenerativeAIEmbeddings(model="embedding-001")
vectorstore = Chroma.from_documents(documents, embedding_model)

# Create Retriever
retriever = vectorstore.as_retriever()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro")
qa_chain = RetrievalQA(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot with Gemini & LangChain")

# Chat Input
user_query = st.text_input("Ask a question about the document:")

if user_query:
    with st.spinner("Generating answer..."):
        response = qa_chain.run(user_query)
        st.write("**Answer:**", response)
