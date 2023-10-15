# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json
import yt_dlp


compute_type="float16"
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model("large-v2", self.device, language="en", compute_type=compute_type)
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)

    def predict(
        self,
        youtube_video_url: str = Input(description="YouTube video URL to transcribe"),
    ) -> str:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            audio_file_path = download_youtube_video_as_m4a(youtube_video_url)
            audio_file = open(audio_file_path, 'rb')
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=32)
        return json.dumps(result['segments'])







def download_youtube_video_as_m4a(url, output_path="."):
    """
    Downloads a YouTube video using the yt_dlp Python package and saves it as an m4a file.

    :param url: The URL of the YouTube video to download.
    :param output_path: Directory where the m4a file will be saved.
    :return: The filename of the downloaded file.
    """

    # This will hold our downloaded filename
    downloaded_filename = None

    def hook(d):
        nonlocal downloaded_filename
        if d['status'] == 'finished':
            downloaded_filename = d['filename']

    options = {
        'format': 'bestaudio[ext=m4a]',  # Get the best m4a audio
        'merge_output_format': 'm4a',    # Ensure the output format is m4a
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Define the output filename format
        'progress_hooks': [hook],  # Add our hook
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([url])

    return downloaded_filename