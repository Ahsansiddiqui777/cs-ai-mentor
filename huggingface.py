from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Flash-0731")


model = ChatHuggingFace(llm=llm)
response = model.invoke("I am Ahsan, and I have recently started my journey into Generative AI.")

print(response.content)