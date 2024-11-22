import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HF_KEY")

from langchain_community.tools import DuckDuckGoSearchResults, ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_groq.chat_models import ChatGroq
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1)

arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddg = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, name="search")

st.title("Search Enging with LLM")

st.sidebar.title("Settings")
groq_api = st.sidebar.text_input("Enter your GROQ API")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": "Hi, I am an AI searchbot who can search the internet, how can I help you?"
    }]

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input():
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api, streaming=True)
    tools = [wiki, arxiv, ddg]
    search_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_callback])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)



