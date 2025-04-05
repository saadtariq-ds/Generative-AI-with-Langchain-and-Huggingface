import streamlit as st
import sqlite3
from pathlib import Path

from keras.src.backend.jax.numpy import absolute
from langchain_groq import ChatGroq
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine

st.set_page_config(page_title="Chat with SQL DB", page_icon=":parrot:")
st.title(":parrot: Chat with SQL DB")

INJECTION_WARNING = """
SQL Agent can be vulnerable to prompt injection. Use a DB role with limited permissions.
Read more [here](https://python.langchain.com/docs/security).
"""

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
mysql_host = ""
mysql_port = ""
mysql_user = ""
mysql_password = ""
mysql_db = ""

radio_options = ['Use Sqllite3 Database','Connect to MySQL Database']
selected_option = st.sidebar.radio(
    label="Choose the DB which you want to chat",
    options=radio_options
)

if radio_options.index(selected_option) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host Name")
    mysql_port = st.sidebar.text_input("Provide MySQL Port Number")
    mysql_user = st.sidebar.text_input("Provide MySQL User Name")
    mysql_password = st.sidebar.text_input("Provide MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("Provide MySQL Database Name")
else:
    db_uri = LOCALDB

api_key = st.sidebar.text_input(label="Provide GROQ API Key", type="password")

if not db_uri:
    st.info("Please enter the database information and uri")

if not api_key:
    st.info("Please add GROQ API Key")

## LLM Model
llm = ChatGroq(groq_api_key=api_key, model_name='llama3-8b-8192', streaming=True)

# DB Configuration
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_port=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_file_path = (Path(__file__).parent/"student.db").absolute()
        print(db_file_path)

        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))

    elif db_uri == MYSQL:
        if not (mysql_host or mysql_user or mysql_password or mysql_db):
            st.error("Please provide all MySQL Connection Details")
            st.stop()

        return SQLDatabase(
            create_engine(
                f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
            )
        )

if db_uri == MYSQL:
    db = configure_db(
        db_uri=db_uri, mysql_host=mysql_host, mysql_user=mysql_user,
        mysql_password=mysql_password, mysql_db=mysql_db, mysql_port=mysql_port
    )
else:
    db = configure_db(db_uri=db_uri)


# Toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state["messages"] = [
        {"role":"assistant", "content":"How can I help you?"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

user_query = st.chat_input(placeholder="Ask from Database")
if user_query:
    st.session_state.messages.append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)