import os

from http.client import responses

from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA



# load environment variables
load_dotenv()

# setup working directory of the current file, to identify the files in the same location
working_dir = os.path.dirname(os.path.abspath(__file__))

#load embedding model
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

embedding = get_embedding()


vec = embedding.embed_query("Hello world")
print(len(vec))


# Load the Llama-3.3-70B model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

def process_document_to_chroma_db(file_name):
    #Load the PDF document using UnstructuredPDFLoader
    # loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    loader = PyPDFLoader(f"{working_dir}/{file_name}")

    documents = loader.load()
    #split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    #store the documents chunks into chroma vector database
    vectordb= Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    # Create a retriever for document view
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    #create a RetrieverQA chain to answer user questions using Llama-3.3-70B

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke({"query": user_question})
    answer = response["result"]


    return answer





