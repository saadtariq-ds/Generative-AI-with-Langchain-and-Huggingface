import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler


st.set_page_config(
    page_title="Text to Math Problem Solver and Data Search Assistant",
    page_icon=":parrot:"
)
st.title("Text to Math Problem Solver and Data Search Assistant")

groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")
if not groq_api_key:
    st.info("Please add your GROQ API Key")
    st.stop()

llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=groq_api_key)

## Initialize Wikipedia Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find various information on the topics mentioned"
)

## Initialize Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided"
)

## Defining Prompt
prompt_template = """ You are a agent tasked to solving users mathematical questions.
Logically arrive at the solution and provide a detailed explanation
and display it in point wise for the question below.
Question: {question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)

## Combine all tools into chain
chain = LLMChain(llm=llm, prompt=prompt)

reasoning_tool = Tool(
    name="Reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions"
)
tools = [wikipedia_tool, math_tool, reasoning_tool]

# Initialize Agents
assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

## Create Session State
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant", "content":"Hi I am a Math Chatbot wo can answer all your maths questions"}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


## Interaction with Streamlit
question = st.text_area("Enter your question")
if st.button("Submit your Question"):
    if question:
        with st.spinner("Generating Response"):
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            st.chat_message("user").write(question)

            streamlit_callback = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=False
            )
            response = assistant_agent.run(
                st.session_state.messages,
                callbacks=[streamlit_callback]
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
            st.write("Response:")
            st.success(response)
    else:
        st.warning("Please Enter the Question")
