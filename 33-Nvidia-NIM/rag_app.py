import os
import streamlit as st
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

## Load Nvidia API Key
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

## Vector Embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader(path="./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=50
        )
        st.session_state.documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30]
        )
        st.session_state.vector_store = FAISS.from_documents(
            documents=st.session_state.documents,
            embedding=st.session_state.embeddings
        )


## Streamlit App
st.title("NVIDIA NIM")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)

query = st.text_input("Enter your Question from Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

if query:
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )
    start_time = time.process_time()
    response = retrieval_chain.invoke(
        {"input":query}
    )
    print(f"Response Time: {time.process_time() - start_time}")
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------------------")