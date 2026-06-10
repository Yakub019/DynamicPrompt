from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
chat_history = []
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
)
model = ChatHuggingFace(llm=llm, temperature=1.0)
while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    chat_history.append(user_input)
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: :", result.content)
    print("Chat History:", chat_history)
    
    