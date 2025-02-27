import whisper
import numpy as np 
import os 
from scipy.io.wavfile import write

class TranscriptionModule():

    def __init__(self, model_size : str = "medium"):

        self.model_size = model_size
        self.model_dir_path = os.path.join(os.path.dirname(__file__), 'models')
        self.model_path = os.path.join(self.model_dir_path, model_size)


        os.makedirs("models", exist_ok = True)

        if not os.path.exists(self.model_path):
            print(f"Model '{model_size}' not found locally. Downloading...")
            self.model = whisper.load_model(model_size, download_root=self.model_path)
        else:
            print(f"Loading existing model '{model_size}' from '{self.model_path}'")
            self.model = whisper.load_model(model_size, download_root=self.model_path)


    def transcribe(self, 
                   audio_array : np.ndarray,
                   temperature : float = 0.0,
                   compression_ratio_threshold : int = 2,
                   condition_on_previous_text : bool = False,
                   ) -> str :
        
        audio_array = audio_array.astype(np.float32)
        rate = 16000
       
        scaled = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
        write('test_processed.wav', rate, scaled)
        transcript = self.model.transcribe(audio_array,
                                           temperature = temperature,
                                           compression_ratio_threshold = compression_ratio_threshold,
                                           condition_on_previous_text = condition_on_previous_text)
        return transcript['text']



