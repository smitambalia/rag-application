import os
import pandas as pd
import json
import re
from tqdm import tqdm
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Embedding model
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def clean_html(text):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def upsert_to_pinecone(chunks, row_id, metadata_dict):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        vector_id = f"{row_id}-{i}"
        # Flatten metadata to simple types
        flat_metadata = {
            "text": chunk,
            "title": metadata_dict.get("title", ""),
            "source": metadata_dict.get("source", ""),
            "content_url": metadata_dict.get("content_url", ""),
            "node_id": metadata_dict.get("node_id", ""),
            "section_number": metadata_dict.get("section", {}).get("number", "") if isinstance(metadata_dict.get("section"), dict) else "",
            "section_heading": metadata_dict.get("section", {}).get("heading", "") if isinstance(metadata_dict.get("section"), dict) else "",
            "part_number": metadata_dict.get("location", {}).get("part_number", "") if isinstance(metadata_dict.get("location"), dict) else "",
            "title_number": metadata_dict.get("location", {}).get("title_number", "") if isinstance(metadata_dict.get("location"), dict) else "",
        }
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": flat_metadata
        })
    # Upsert in batches (max 100 per Pinecone API)
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

def main():
    csv_file = 'ecfr.csv'
    df = pd.read_csv(csv_file)
    print(f"Processing {len(df)} rows from {csv_file}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            content = row['content']
            title = row['title']
            source = row['source']
            metadata_str = row['metadata']
            # Handle NaN values
            if pd.isna(content):
                content = ""
            if pd.isna(title):
                title = ""
            if pd.isna(source):
                source = ""
            metadata_dict = {}
            if not pd.isna(metadata_str) and isinstance(metadata_str, str):
                try:
                    metadata_dict = json.loads(metadata_str)
                except json.JSONDecodeError:
                    metadata_dict = {}
            # Clean HTML from content
            cleaned_content = clean_html(str(content))
            # Combine title and cleaned content for chunking
            text_to_chunk = f"{title} {cleaned_content}".strip()
            if not text_to_chunk:
                continue  # Skip empty rows
            chunks = chunk_text(text_to_chunk)
            # Prepare metadata for Pinecone
            pinecone_metadata = {
                "title": title,
                "source": source,
                "content_url": row.get('content_url', ''),
                **metadata_dict
            }
            upsert_to_pinecone(chunks, row['id'], pinecone_metadata)
        except Exception as e:
            print(f"Error processing row {idx} (ID: {row['id']}): {e}")

if __name__ == "__main__":
    main()