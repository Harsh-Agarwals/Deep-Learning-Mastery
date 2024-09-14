import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import random

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLEAI_API_KEY")

# LLM = ChatOpenAI(
#     openai_api_key=OPENAI_KEY,
#     model_name="gpt-3.5-turbo",
#     temperature=0.2,
#     max_tokens=256,
#     max_retries=2
# )

LLM = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a conversation bot capable of doing all language tasks"),
        ("human", "{input}")
    ]
)

output_parser = StrOutputParser()

chain = prompt|LLM|output_parser

st.title("ChatGPT clone")

i=0
def conversation():
    user_query = st.text_input(label="Your query", key=random.randint(0, 1000))
    result = st.button(label="Submit", key=random.randint(0, 1000))
    if result:
        response = chain.invoke({"input": user_query})
        st.write(response)
        conversation()

conversation()