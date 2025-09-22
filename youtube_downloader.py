import yt_dlp
import streamlit as st

def get_available_formats(url):
    """
    Fetches available video formats for a given YouTube URL without downloading.
    Returns a list of format dictionaries.
    """
    ydl_opts = {'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            # Filter for formats that have both video and audio, are mp4, and sort by resolution
            formats = [
                f for f in info.get('formats', []) 
                if f.get('vcodec') != 'none' 
                and f.get('acodec') != 'none' 
                and f.get('ext') == 'mp4'
            ]
            # Sort by height (resolution) in descending order
            return sorted(formats, key=lambda x: x.get('height', 0), reverse=True)
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Error fetching video info: Please check the URL.")
            return []

def download_video(url, format_id, output_path_template, progress_bar):
    """
    Downloads a specific video format and updates a Streamlit progress bar.
    Returns the final path of the downloaded file.
    """
    final_path = ""

    def postprocessor_hook(d):
        nonlocal final_path
        if d['status'] == 'finished':
            final_path = d['info_dict']['filepath']

    def progress_hook(d):
        if d['status'] == 'downloading':
            # Use total_bytes_estimate if available, otherwise fall back to total_bytes
            total_bytes = d.get('total_bytes_estimate') or d.get('total_bytes')
            if total_bytes:
                percent = d['downloaded_bytes'] / total_bytes
                progress_bar.progress(percent, text=f"Downloading... {int(percent*100)}%")
        if d['status'] == 'finished':
            progress_bar.progress(1.0, text="Download complete! Finalizing...")

    ydl_opts = {
        'format': format_id,
        'outtmpl': output_path_template,
        'progress_hooks': [progress_hook],
        'postprocessor_hooks': [postprocessor_hook],
        'quiet': True,
        'merge_output_format': 'mp4', # Ensuring output is mp4
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return final_path
        except yt_dlp.utils.DownloadError as e:
            st.error(f"Error during download: {e}")
            return None