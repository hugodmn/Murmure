import librosa 
from typing import Tuple
import numpy as np 


def load_audio(audio_path : str) -> Tuple[np.ndarray, int]:

    audio, sr = librosa.load(audio_path)

    return audio, sr