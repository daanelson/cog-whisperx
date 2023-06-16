# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ["HF_HOME"] = "/src/hf_models"
os.environ["TORCH_HOME"] = "/src/torch_models"
from typing import Any

from cog import BasePredictor, Input, Path
from whisper_jax import FlaxWhisperPipline


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = FlaxWhisperPipline(
            "vasista22/whisper-hindi-large-v2", batch_size=16
        )
        self.pipe.model.config.forced_decoder_ids = (
            self.pipe.tokenizer.get_decoder_prompt_ids(language="hi", task="transcribe")
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
    ) -> Any:
        print("transcribing")
        audio = str(audio)
        transcription = self.pipe(audio)

        return transcription["text"]
