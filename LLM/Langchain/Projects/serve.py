import os
from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.3,
    max_tokens=256,
    api_key=GROQ_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an AI system that makes {genre} story from the key provided by the user"),
    HumanMessage(content="{key}")
])

parser = StrOutputParser()

chain = prompt | model | parser
