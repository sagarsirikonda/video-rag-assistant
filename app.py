# app.py

import streamlit as st
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# Import your backend modules
from transcriber import process_video, load_model as load_whisper_model
from vector_store import create_vector_store, load_embedding_model
from rag_pipeline import create_rag_chain, parse_llm_response

# --- LLM and LangChain Imports ---
from langchain_groq import ChatGroq

# --- Page Configuration ---
st.set_page_config(
    page_title="Video RAG Assistant",
    page_icon="🎥",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Caching Models for Performance ---
@st.cache_resource
def get_whisper_model(): return load_whisper_model("base")
@st.cache_resource
def get_embedding_model(): return load_embedding_model()
@st.cache_resource
def get_llm(): return ChatGroq(model_name="llama3-8b-8192", temperature=0)

# --- Helper Function for Timestamp Conversion ---
def timestamp_to_seconds(timestamp_str):
    match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})-(\d{2}):(\d{2}):(\d{2})\]', timestamp_str)
    if match:
        h, m, s = map(int, match.groups()[:3])
        return h * 3600 + m * 60 + s
    return 0

# --- Main Application Logic ---
def main():
    st.title("🎥 Video RAG Assistant")

    # --- Session State Initialization ---
    if "messages" not in st.session_state: st.session_state.messages = []
    if "video_processed" not in st.session_state: st.session_state.video_processed = False
    if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
    if "video_path" not in st.session_state: st.session_state.video_path = None
    if "clicked_timestamps" not in st.session_state: st.session_state.clicked_timestamps = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("Upload Video")
        uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
        
        # --- NEW: Clear Chat Button ---
        if st.session_state.video_path:
            st.markdown("---")
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.clicked_timestamps = []
                st.rerun()

        if uploaded_file is not None:
            video_dir = "data/videos"
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, uploaded_file.name)
            with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())

            if st.session_state.video_path != video_path:
                st.session_state.video_path = video_path
                st.session_state.video_processed = False
                st.session_state.messages = []
                st.session_state.clicked_timestamps = []

            if not st.session_state.video_processed:
                # --- NEW: Detailed Progress Updates ---
                progress_placeholder = st.empty()
                
                progress_placeholder.text("⏳ Step 1/3: Generating transcript...")
                transcripts_dir = "data/transcripts"
                os.makedirs(transcripts_dir, exist_ok=True)
                transcript_path = process_video(video_path, transcripts_dir, get_whisper_model())
                
                progress_placeholder.text("🧠 Step 2/3: Creating vector store...")
                video_filename = Path(video_path).stem
                vector_store_dir = os.path.join("data", "vector_store", video_filename)
                create_vector_store(transcript_path, vector_store_dir, get_embedding_model())
                
                progress_placeholder.text("🔗 Step 3/3: Building RAG chain...")
                st.session_state.rag_chain = create_rag_chain(vector_store_dir, get_llm())
                
                st.session_state.video_processed = True
                progress_placeholder.empty() # Clear the progress text
                st.success("Video processed successfully!")
                time.sleep(2) # Give user time to read the success message
                st.rerun()
    
    # --- Main Content Area ---
    if st.session_state.video_path:
        # (Chat Interface and Video Clip display logic remains the same)
        for msg_index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamps" in message and message["timestamps"]:
                    cols = st.columns(len(message["timestamps"]))
                    for ts_index, ts in enumerate(message["timestamps"]):
                        unique_key = f"ts_{msg_index}_{ts_index}"
                        if cols[ts_index].button(ts, key=unique_key):
                            if ts not in st.session_state.clicked_timestamps:
                                st.session_state.clicked_timestamps.append(ts)
                            st.rerun()

        if st.session_state.clicked_timestamps:
            st.markdown("---")
            st.subheader("Video Clips")
            for ts in st.session_state.clicked_timestamps:
                start_time = timestamp_to_seconds(ts)
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.video(st.session_state.video_path, start_time=start_time)

        if prompt := st.chat_input("Ask a question about the video..."):
            if not st.session_state.video_processed:
                st.warning("Please wait for the video to be processed.")
            else:
                st.session_state.clicked_timestamps = []
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_chain.invoke(prompt)
                    answer, timestamps = parse_llm_response(response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamps": timestamps
                    })
                    st.rerun()
    else:
        st.info("Please upload a video file in the sidebar to get started.")

if __name__ == "__main__":
    main()