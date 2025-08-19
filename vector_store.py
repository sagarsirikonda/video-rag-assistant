# vector_store.py

import os
import json
from datetime import timedelta
# --- LangChain Imports ---
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# (The format_timestamp and create_chunks functions remain the same)
def format_timestamp(seconds: float) -> str:
    """Formats seconds into HH:MM:SS format."""
    td = timedelta(seconds=round(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    """Loads the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)

def create_chunks(segments: list[dict], chunk_size: int = 10) -> list[dict]:
    """
    Creates text chunks from transcript segments by grouping a fixed number of segments.
    """
    chunks = []
    if not segments:
        return []

    for i in range(0, len(segments), chunk_size):
        chunk_segments = segments[i:i + chunk_size]
        combined_text = " ".join(seg['text'].strip() for seg in chunk_segments)
        start_time = chunk_segments[0]['start']
        end_time = chunk_segments[-1]['end']
        
        chunks.append({
            "text": combined_text,
            "timestamp": f"[{format_timestamp(start_time)}-{format_timestamp(end_time)}]"
        })
        
    return chunks

# --- REWRITTEN AND IMPROVED VECTOR STORE CREATION FUNCTION ---
def create_vector_store(transcript_path: str, vector_store_dir: str, embedding_model):
    """
    Creates and saves a FAISS vector store using the modern LangChain method.

    Args:
        transcript_path (str): The path to the transcript JSON file.
        vector_store_dir (str): The directory to save the FAISS index.
        embedding_model: The loaded HuggingFaceEmbeddings instance.
    """
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    # Load and chunk the transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_segments = json.load(f)
    
    chunks = create_chunks(transcript_segments)
    
    # Separate the texts and their corresponding metadata (timestamps)
    chunk_texts = [chunk['text'] for chunk in chunks]
    metadatas = [{"timestamp": chunk['timestamp']} for chunk in chunks]

    # Create the FAISS vector store from the texts and metadatas
    # This single command handles embedding, indexing, and structuring the data.
    vectorstore = FAISS.from_texts(
        texts=chunk_texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    # Save the vector store locally. This will create index.faiss and index.pkl
    os.makedirs(vector_store_dir, exist_ok=True)
    vectorstore.save_local(vector_store_dir)
    
    print(f"Vector store created and saved at: {vector_store_dir}")
    return vector_store_dir

# This block allows for testing the script directly
if __name__ == '__main__':
    TEST_TRANSCRIPT_PATH = "data/transcripts/The ginormous collision that tilted our planet - Elise Cutts [vCbx5jtZ_qI]_transcript.json"
    
    VIDEO_FILENAME = os.path.splitext(os.path.basename(TEST_TRANSCRIPT_PATH))[0].replace("_transcript", "")
    VECTOR_STORE_SAVE_DIR = os.path.join("data", "vector_store", VIDEO_FILENAME)

    if not os.path.exists(TEST_TRANSCRIPT_PATH):
        print("="*50)
        print(f"TESTING SKIPPED: Please update 'TEST_TRANSCRIPT_PATH' in vector_store.py")
        print(f"Current path is: {TEST_TRANSCRIPT_PATH}")
        print("="*50)
    else:
        print("Loading embedding model...")
        # We now load the model here to pass it to the creation function
        embeddings = load_embedding_model()

        print(f"Starting vector store creation for: {TEST_TRANSCRIPT_PATH}")
        create_vector_store(
            transcript_path=TEST_TRANSCRIPT_PATH,
            vector_store_dir=VECTOR_STORE_SAVE_DIR,
            embedding_model=embeddings
        )