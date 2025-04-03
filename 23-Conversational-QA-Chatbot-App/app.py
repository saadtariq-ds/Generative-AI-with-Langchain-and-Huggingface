import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"] = "Conversational QA Chatbot"

llm = ChatGroq(model_name='llama3-8b-8192')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Streamlit App
st.title("Conversational QA Chatbot with Chat History")
st.write("Upload PDFs and Chat with their content")

api_key = st.text_input("Enter your GROQ API Key:", type="password")

if api_key:
    llm = ChatGroq(model_name='llama3-8b-8192')

    # Chat Interface
    session_id = st.text_input("Session ID", value='default_session')

    # Manage Chat History
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose a PDF file", type='pdf', accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(file=temp_pdf, mode='wb') as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Embeddings and Vector Store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, chunk_overlap=500
        )
        splits = text_splitter.split_documents(documents=documents)
        vector_store = Chroma.from_documents(
            documents=splits, embedding=embeddings
        )
        retriever = vector_store.as_retriever()

        contextualize_system_prompt = """ Given a chat history and latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood without chat history.
        Do not answer the question, just reformulate it if needed and otherwise return it as is
        """

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm=llm, retriever=retriever, prompt=contextualize_prompt
        )

        # Answer Question Prompt
        system_prompt = """ You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, say that you don't know.
        Use three sentences maximum and keep the answer concise
        \n\n
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(
            llm=llm, prompt=qa_prompt
        )

        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever, combine_docs_chain=question_answer_chain
        )

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history=get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

        user_input = st.text_input("Your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable" : {"session_id":session_id}
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)

else:
    st.warning("Please enter GROQ API")