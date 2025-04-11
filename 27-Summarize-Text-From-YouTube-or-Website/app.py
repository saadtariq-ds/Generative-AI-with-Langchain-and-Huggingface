import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit App
st.set_page_config(
    page_title="LangChain: Summarize Text from Website or YouTube",
    page_icon=":parrot:"
)
st.title(":parrot: LangChain: Summarize Text from Website or YouTube")
st.subheader("Summarize URL")

## Get Groq API Key and URL Field to Summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Setting LLM
llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)

## Setting Prompt Template
prompt_template = """ Provide a summary of the following content in 300 words:
Content: {text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

if st.button("Summarize the Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the Information")
    elif not validators.url(generic_url):
        st.error("Please enter a Valid URL from Website or YouTube")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(
                        youtube_url=generic_url, add_video_info=True
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], ssl_verify=False,
                        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"}
                    )
                data = loader.load()

                ## Chain for Summarization
                chain = load_summarize_chain(
                    llm=llm, chain_type='stuff', prompt=prompt
                )
                output_summary = chain.run(data)
                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")

