import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek")  # or "llama2", etc.


# Load the embedding model once at startup
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

app = Flask(__name__)

def ask_ollama(prompt):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]


def get_embedding(text):
    # Returns a list (vector) for Pinecone
    return embedding_model.encode(text).tolist()


def query_pinecone(embedding, top_k=3):
    result = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return result['matches']

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    query = data.get('query', '')

    embedding = get_embedding(query)
    matches = query_pinecone(embedding)
    context = "\n".join([m['metadata'].get('text', '') for m in matches])

    # answer = f"Context:\n{context}\n\nYou asked: {query}. (LLM answer will go here.)"
    # Build the prompt for the LLM
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = ask_ollama(prompt)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)