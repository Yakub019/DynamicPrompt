import warnings
warnings.filterwarnings("ignore")
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm = llm , temperature = 1.0)
chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ", result.content)

print("chat_history:",chat_history)
