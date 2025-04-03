import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"] = "QA Generative AI Chatbot App"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("human", "Question:{question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    api_key=api_key
    llm=ChatOpenAI(model_name=llm, openai_api_key=api_key,
                   temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer


# Streamlit App
st.title("QA Chatbot")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
llm = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4o", "gpt-4-turbo", "gpt-4"])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Ask any Question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(question=user_input, api_key=api_key, llm=llm,
                                 temperature=temperature, max_tokens=max_tokens)
    st.write(response)
