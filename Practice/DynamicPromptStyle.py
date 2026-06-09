from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from typer import prompt
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm, temperature=1.0)
paper_input = st.selectbox("Select Research Paper Name",["Attention is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"])
style_input = st.selectbox("Select Explanation Style",["Begginer friendly","Technical","code oriented","Mathematical"])
Length_input = st.selectbox("Select Explanation Length",["Short 1-2 paragraphs","Medium 3-5 paragraphs","Long detailed explanation"])
Template=PromptTemplate(
template = """Please Summarize the research paper titled {paper_input} with following specifications:
Explanation Style: {style_input}
Explanation Length: {Length_input}
1. Mathematical details:
- include relevant equations and formulas from the paper.
- provide explanations for each equation
2.analogies and examples:
-use analogies and examples to explain complex concepts in the paper.
if certain is not available in the paper respond with "Not Available in the paper"
ensure that the summary is clear, concise, and easy to understand for the specified explanation style and length""",
input_variables=["paper_input","style_input","Length_input"]
)
prompt=Template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "Length_input": Length_input
})
user_input = st.text_input("Prompt")
st.header("Dynamic Prompt")
if st.button ('Summarize'):
    result=model.invoke(prompt)
    st.write(result.content)