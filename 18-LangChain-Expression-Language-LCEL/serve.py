import os
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="Gemma2-9b-It")

## Chat Prompt Template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user","{text}")
    ]
)

## Parser
parser = StrOutputParser()

## Create Chain
chain = prompt_template | llm | parser

## App Definition
app = FastAPI(
    title='LangChain Server',
    version="1.0",
    description="A simple API Server using LangChain Runnable Interfaces")

add_routes(
    app,
    chain,
    path='/chain'
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host='localhost', port=8000)