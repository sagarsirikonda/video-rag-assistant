import whisper
import os
import subprocess
import json
from pathlib import Path

# Note: The model loading is wrapped in a function.
# In our Streamlit app (app.py), we will use @st.cache_resource
# to load this model only once.
def load_model(model_name="base"):
    """Loads the Whisper model."""
    return whisper.load_model(model_name)

def extract_audio(video_path: str, audio_path: str):
    """
    Extracts the audio from a video file using FFmpeg.

    Args:
        video_path (str): The path to the input video file.
        audio_path (str): The path where the output audio file will be saved.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    # --- UPDATED COMMAND ---
    # This command is more robust:
    # -vn: No video output
    # -c:a mp3: Explicitly set the audio codec to mp3
    # -q:a 2: Set a high-quality variable bitrate (VBR)
    # -y: Overwrite the output file if it exists
    command = f'ffmpeg -i "{video_path}" "{audio_path}" -y'
    try:
        # Using DEVNULL to hide ffmpeg's console output for a cleaner UX
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting audio: {e}")
        # To see the detailed error from ffmpeg, you can temporarily change the
        # subprocess.run call to: subprocess.run(command, shell=True, check=True)
        raise

def generate_transcript(audio_path: str, model) -> list[dict]:
    """
    Generates a transcript from an audio file using the loaded Whisper model,
    and cleans it to keep only essential information.

    Args:
        audio_path (str): The path to the input audio file.
        model: The loaded Whisper model instance.

    Returns:
        list[dict]: A list of simplified segment dictionaries with only
                    'start', 'end', and 'text'.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    # Perform the transcription
    result = model.transcribe(audio_path, verbose=False)
    
    # --- UPDATED SECTION TO CLEAN JSON ---
    essential_segments = []
    for segment in result["segments"]:
        essential_segments.append({
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "text": segment["text"]
        })
    
    return essential_segments

def process_video(video_path: str, output_dir: str, model) -> str:
    """
    The main processing pipeline for a single video.
    1. Extracts audio.
    2. Generates transcript.
    3. Saves transcript as a JSON file.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save audio and transcript files.
        model: The loaded Whisper model instance.

    Returns:
        str: The path to the saved transcript JSON file.
    """
    video_filename = Path(video_path).stem
    
    # Define paths for the audio and transcript files
    audio_path = os.path.join(output_dir, f"{video_filename}.mp3")
    transcript_path = os.path.join(output_dir, f"{video_filename}_transcript.json")

    # Step 1: Extract audio from the video
    extract_audio(video_path, audio_path)

    # Step 2: Generate the transcript
    transcript_segments = generate_transcript(audio_path, model)

    # Step 3: Save the transcript to a JSON file
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript_segments, f, indent=4)

    # Clean up the temporary audio file
    os.remove(audio_path)

    return transcript_path

# This block allows for testing the script directly
if __name__ == '__main__':
    # --- Configuration for Testing ---
    TEST_VIDEO_PATH = "C:/Users/sirik/OneDrive/Desktop/Siri/The ginormous collision that tilted our planet - Elise Cutts [vCbx5jtZ_qI].mp4" # IMPORTANT: Change this to a real video file
    
    # We'll use our planned data structure for the test
    DATA_DIR = "data"
    TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
    
    # Create directories if they don't exist
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    
    if not os.path.exists(TEST_VIDEO_PATH):
        print("="*50)
        print(f"TESTING SKIPPED: Please update 'TEST_VIDEO_PATH' in transcriber.py")
        print(f"Current path is: {TEST_VIDEO_PATH}")
        print("="*50)
    else:
        print("Loading Whisper model...")
        whisper_model = load_model("base")
        
        print(f"Starting processing for: {TEST_VIDEO_PATH}")
        try:
            final_transcript_path = process_video(
                video_path=TEST_VIDEO_PATH,
                output_dir=TRANSCRIPTS_DIR,
                model=whisper_model
            )
            print("-" * 50)
            print("Processing complete!")
            print(f"Transcript saved at: {final_transcript_path}")
            print("-" * 50)
            
            # Print first 5 segments as a preview
            with open(final_transcript_path, 'r') as f:
                data = json.load(f)
                print("Transcript preview (first 5 segments):")
                for segment in data[:5]:
                    start = round(segment['start'], 2)
                    end = round(segment['end'], 2)
                    text = segment['text']
                    print(f"[{start:.2f}s - {end:.2f}s]: {text}")
                    
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")