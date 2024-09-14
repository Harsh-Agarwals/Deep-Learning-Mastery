import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLEAI_API_KEY")

LLM = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful translator which translates from {input_language} to {output_language}."),
        ("human", "{input}")
    ]
)

output_parser = StrOutputParser()

chain = prompt|LLM|output_parser

st.title("Language translation with Google Gemini")
user_text = st.text_input("Enter a text that will translate from english to hindi")
result = st.button("Submit")

if result:
    response = chain.invoke({
        "input_language": "english",
        "output_language": "hindi",
        "input": user_text
    })
    st.write(response)