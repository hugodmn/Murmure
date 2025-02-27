import os 
import torch 
import urllib.request
import numpy as np 
from silero_vad import get_speech_timestamps
from typing import Dict, List

class VADModule():

    def __init__(self,
                 device :str = 'cpu'):

        self.model_dir_path = os.path.join(os.path.dirname(__file__), 'models')
        self.model_path = os.path.join(self.model_dir_path,"silero_vad.jit")

        print(f"Loading Silero VAD model from '{self.model_path}'")
        self.model = torch.jit.load(self.model_path)
        self.model.eval()

    def detect_speech_segments(self,
                               audio : np.ndarray,
                               threshold : float = 0.2,
                               neg_threshold : float = 0.02,
                               min_silence_duration_ms = 3000,
                               min_speech_duration_ms = 250,

                               ) :
        
        speech_timestamps = get_speech_timestamps(
            model = self.model,
            audio = audio, 
            threshold = threshold,
            neg_threshold = neg_threshold,
            min_silence_duration_ms = min_silence_duration_ms,
            min_speech_duration_ms = min_speech_duration_ms,
            )
        
        return speech_timestamps
    

    def segmentation(self,
                     audio : np.ndarray,
                     speech_timestamps : List[Dict],
                     sr : int = 16000,
                     overlap_ms : int = 200,
                     padding_ms : int = 1000) :
        
        segmented_audio = np.array([])
        for segment in speech_timestamps :

            begin = int(segment['start']) - int((overlap_ms/1000)*sr)
            if begin < 0 : 
                begin = 0 
         
            end = int(segment['end']) + int((overlap_ms/1000)*sr)
        
            if int(len(audio)) - end < 0:
                end = int(len(audio)) 
           
            audio_slice = audio[begin : end]

            segmented_audio = np.concatenate((segmented_audio, audio_slice))

        return segmented_audio           




