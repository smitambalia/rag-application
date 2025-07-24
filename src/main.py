import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
import time
from groq import Groq


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek")  # or "llama2", etc.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # 'openai', 'groq', or 'ollama'

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Initialize OpenAI and Groq clients
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

def ask_groq(prompt):
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

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
    return response.json().response

# def ask_openai(prompt):
#     response = openai_client.chat.completions.create(
#         model=OPENAI_MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#         max_tokens=512
#     )
#     return response.choices[0].message.content.strip()


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

    timings = {}

    start = time.time()
    embedding = get_embedding(query)
    timings['embedding_time'] = time.time() - start

    start = time.time()
    matches = query_pinecone(embedding)
    timings['query_pinecone_time'] = time.time() - start

    context = "\n".join([m['metadata'].get('text', '') for m in matches])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    start = time.time()
  
    if LLM_PROVIDER == "groq":
        answer = ask_groq(prompt)
    else:
        answer = ask_ollama(prompt)
    timings['ask_llm_time'] = time.time() - start

    return jsonify({
        'answer': answer,
        'timings': timings
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)