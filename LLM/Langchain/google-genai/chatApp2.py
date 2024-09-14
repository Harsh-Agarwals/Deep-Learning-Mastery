import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLEAI_API_KEY")

llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-1.5-flash",
    max_tokens=None,
    temperature=0,
    max_retries=2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot"),
    ("human", "question: {question}")
])

output_parser = StrOutputParser()

chain = prompt|llm|output_parser

st.title("Google Generative AI App")
user_query = st.text_input("Ask question to Google Gemini")
result = st.button("Submit")

if result:
    st.write(chain.invoke({"question": user_query}))