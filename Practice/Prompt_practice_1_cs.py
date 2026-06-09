""" static prompt are sensitive because it more depended on prompt,
less information in prompt leads to haloginate or diffent information """
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
st.header("Static Prompt")
user_input = st.text_input("enter your Prompt")
if st.button("summarize"):
    result=model.invoke(user_input)
    st.write(result)
