import os
import streamlit as st
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import initialize_agent, AgentType, create_openai_tools_agent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

load_dotenv()

## Creating Tools
api_wrapper_wikipedia = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper_wikipedia)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search_tool = DuckDuckGoSearchRun(name="Search")


st.title("LangChain - Tools and Agents")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role":"assistant",
            "content": "Hi, I am a chatbot who can search the web. How can I help you?"
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model_name='llama3-8b-8192', groq_api_key=api_key, streaming=True)

    tools = [search_tool, wikipedia_tool, arxiv_tool]

    search_agent = initialize_agent(
        tools=tools, llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(
            st.session_state.messages, callbacks=[streamlit_callback]
        )
        st.session_state.messages.append(
            {"role":"assistant", "content":response}
        )
        st.write(response)
