import streamlit as st
import os
import time
import re
from pathlib import Path
from dotenv import load_dotenv

# Import our modules
from transcriber import process_video, load_model as load_whisper_model
from vector_store import create_vector_store, load_embedding_model
from rag_pipeline import create_rag_chain, parse_llm_response
import youtube_downloader

# --- LLM and LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

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
def get_llm():
    # To switch models, just uncomment the line you want to use.
    # Make sure the correct API key is in your .env file.

    # Groq (Default)
    # llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    # Gemini (Uncomment to use)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                temperature=0,
                                google_api_key=os.getenv("GEMINI_API_KEY")
                                )
    
    # OpenAI (Uncomment to use)
    # llm = ChatOpenAI(model="gpt-3.5-turbo", 
    #                  temperature=0, 
    #                  api_key=os.getenv("OPENAI_API_KEY")
    #                  )
    return llm

# --- Helper Functions ---
def timestamp_to_seconds(timestamp_str):
    match = re.match(r'\[(\d{2}):(\d{2}):(\d{2})-(\d{2}):(\d{2}):(\d{2})\]', timestamp_str)
    if match:
        h, m, s = map(int, match.groups()[:3])
        return h * 3600 + m * 60 + s
    return 0

# --- NEW: Function to Sanitize Filenames ---
def sanitize_filename(filename):
    """
    Removes illegal characters from a filename by whitelisting safe characters.
    """
    return re.sub(r'[^a-zA-Z0-9\s._-]', '', filename)

# --- Main Application Logic ---
def main():
    st.title("🎥 Video RAG Assistant")
    

    # --- Session State Initialization ---
    if "messages" not in st.session_state: st.session_state.messages = []
    if "video_processed" not in st.session_state: st.session_state.video_processed = False
    if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
    if "video_path" not in st.session_state: st.session_state.video_path = None
    if "clicked_timestamps" not in st.session_state: st.session_state.clicked_timestamps = []
    if "formats" not in st.session_state: st.session_state.formats = []

    # --- Sidebar ---
    with st.sidebar:
        st.header("Video Source")
        source_option = st.radio("Choose source:", ("Upload Local File", "Download from YouTube"))

        if st.session_state.video_path:
            st.markdown("---")
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.session_state.clicked_timestamps = []
                st.rerun()

        # --- Logic for File Upload ---
        if source_option == "Upload Local File":
            uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                video_dir = "data/videos"
                os.makedirs(video_dir, exist_ok=True)
                
                # --- Sanitizing the uploaded filename ---
                sanitized_name = sanitize_filename(uploaded_file.name)
                video_path = os.path.join(video_dir, sanitized_name)
                
                with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                if st.session_state.video_path != video_path:
                    st.session_state.video_path = video_path
                    st.session_state.video_processed = False
                    st.session_state.messages = []
                    st.session_state.clicked_timestamps = []

        # --- Logic for YouTube Download ---
        if source_option == "Download from YouTube":
            youtube_url = st.text_input("Paste YouTube URL here:", key="youtube_url_input")
            if st.button("Get Video Qualities", key="get_qualities"):
                with st.spinner("Fetching available formats..."):
                    st.session_state.formats = youtube_downloader.get_available_formats(youtube_url)
            
            if st.session_state.formats:
                st.subheader("Available Qualities")
                progress_placeholder = st.empty()
                for f in st.session_state.formats:
                    resolution = f.get('format_note', f.get('resolution', 'N/A'))
                    filesize_mb = f.get('filesize', 0) / (1024*1024) if f.get('filesize') else 'N/A'
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if isinstance(filesize_mb, (int, float)):
                            st.write(f"{resolution}, ~{filesize_mb:.2f} MB")
                        else:
                            st.write(f"{resolution}, Size: {filesize_mb}")
                    with col2:
                        if st.button("Download", key=f.get('format_id')):
                            with st.spinner("Downloading video..."):
                                video_dir = "data/videos"
                                os.makedirs(video_dir, exist_ok=True)
                                output_template = os.path.join(video_dir, '%(title)s.%(ext)s')
                                progress_bar = progress_placeholder.progress(0, "Starting Download...")
                                
                                downloaded_path = youtube_downloader.download_video(youtube_url, f.get('format_id'), output_template, progress_bar)
                                
                                if downloaded_path and os.path.exists(downloaded_path):
                                    # --- Sanitizing the downloaded filename ---
                                    base_dir = os.path.dirname(downloaded_path)
                                    original_filename = os.path.basename(downloaded_path)
                                    sanitized_filename = sanitize_filename(original_filename)
                                    sanitized_path = os.path.join(base_dir, sanitized_filename)
                                    
                                    if os.path.exists(sanitized_path):
                                        os.remove(downloaded_path)
                                        st.session_state.video_path = sanitized_path
                                    else:
                                    # Rename the file on disk
                                        os.rename(downloaded_path, sanitized_path)
                                    
                                        st.session_state.video_path = sanitized_path 
                                        st.session_state.video_processed = False
                                        st.session_state.messages = []
                                        st.session_state.clicked_timestamps = []
                                        st.session_state.formats = []
                                        progress_placeholder.empty()
                                        st.success("Download complete! Processing will begin.")
                                        time.sleep(2)
                                        st.rerun()

    # --- Video Processing Logic ---
    if st.session_state.video_path and not st.session_state.video_processed:
        with st.spinner("Processing video..."):
            transcripts_dir = "data/transcripts"
            os.makedirs(transcripts_dir, exist_ok=True)
            transcript_path = process_video(st.session_state.video_path, transcripts_dir, get_whisper_model())
            video_filename = Path(st.session_state.video_path).stem
            vector_store_dir = os.path.join("data", "vector_store", video_filename)
            create_vector_store(transcript_path, vector_store_dir, get_embedding_model())
            st.session_state.rag_chain = create_rag_chain(vector_store_dir, get_llm())
            st.session_state.video_processed = True
        st.success("Video processed and ready!")
        st.rerun()

    # --- Main Content Area ---
    if st.session_state.video_processed:
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
    elif not source_option == "Download from YouTube":
        st.info("Please upload a local video or download one from YouTube to get started.")

if __name__ == "__main__":
    main()