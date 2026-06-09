from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm, temperature=1.0)
import streamlit as st
user_input = st.text_input("Prompt")
if st.button('Submit'):
    result=model.invoke(user_input)
    st.write(result.content)