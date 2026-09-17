from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector = embeddings.embed_query("You are going to learn Gen AI")

print(vector)