import os
import boto3
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS S3 setup
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Embedding model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def list_s3_files(bucket):
    response = s3.list_objects_v2(Bucket=bucket)
    return [obj['Key'] for obj in response.get('Contents', [])]

def get_s3_file_content(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj['Body'].read().decode('utf-8')

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def upsert_to_pinecone(chunks, file_key):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vector_id = f"{file_key}-{i}"
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk, "source": file_key}
        })
    # Upsert in batches (max 100 per Pinecone API)
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

def main():
    files = list_s3_files(S3_BUCKET)
    print(f"Found {len(files)} files in S3 bucket '{S3_BUCKET}'")
    for file_key in tqdm(files):
        try:
            text = get_s3_file_content(S3_BUCKET, file_key)
            chunks = chunk_text(text)
            upsert_to_pinecone(chunks, file_key)
        except Exception as e:
            print(f"Error processing {file_key}: {e}")

if __name__ == "__main__":
    main()