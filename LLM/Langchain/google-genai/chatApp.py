from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

GOOGLEAI_API_KEY = os.getenv("GOOGLEAI_API_KEY")
print("Google AI API Key: ", GOOGLEAI_API_KEY)

st.title("Chat with Google Gemini AI")
user_query = st.text_input("Ask question to Google Gemini")
result = st.button("Submit")

if result:
    st.subheader(user_query)
    try:
        llm = ChatGoogleGenerativeAI(google_api_key=GOOGLEAI_API_KEY, model="gemini-1.5-flash")
        response = llm.invoke(user_query)
        st.write(response.content)
    except:
        st.write("Error loading answer")
