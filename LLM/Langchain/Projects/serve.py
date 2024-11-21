import os
import uvicorn
from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Story Generator App",
    version="1.0.0",
    description="Provide a genre with a topic, and this will output a small story based on topic for this genre."
)

model = ChatGroq(
    model="gemma2-9b-it",
    temperature=0.3,
    max_tokens=256,
    api_key=GROQ_API_KEY
)

# prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content="You are an AI system that makes {genre} story from the key provided by the user"),
#     HumanMessage(content="{key}")
# ])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI system that makes {genre} story from the key provided by the user"),
    ("user", "{key}")
])

parser = StrOutputParser()

chain = prompt | model | parser

# FastAPI routes 
add_routes(
    app,
    chain,
    path="/story"
)

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

