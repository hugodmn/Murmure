import os 
import torch 
import urllib.request
import numpy as np 
from silero_vad import get_speech_timestamps
from typing import Dict, List, Tuple 
from .types import VADSegmentOutput


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
                               threshold : float = 0.3,
                               neg_threshold : float = 0.05,
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
            return_seconds = True,
            )
        

        
        return speech_timestamps
    

    def segmentation(self,
                     audio : np.ndarray,
                     speech_timestamps : List[Dict],
                     sr : int = 16000,
                     overlap_ms : int = 200,
                     padding_ms : int = 200) -> Tuple[np.ndarray, List[dict]]:
        
        segmented_audio = np.array([])
        end = 0 
        last_end = 0
        silence_signal_removed = 0 
        vad_audio_timestamps = list()

        for segment in speech_timestamps :

            last_end = end 

            start = float((segment['start']) - float((overlap_ms/1000)))
            if start < 0 : 
                start = 0 
         
            end = float((segment['end']) + float((overlap_ms/1000)))
            

            if int(audio.size) - end < 0:
                end = int(len(audio)) 
           

            audio_slice = audio[int(start*sr) : int(end*sr)]


            silence_signal_removed = silence_signal_removed + start - last_end

            # segmented_audio_info.append({
            #     "vad_segment_idx" : len(segmented_audio_info) + 1, 
            #     "silence_signal_removed" : silence_signal_removed,
            #     "start": segmented_audio.size,
            #     "end" : segmented_audio.size + audio_slice.size,
            # })

            vad_audio_timestamps.append(VADSegmentOutput(
                idx=len(vad_audio_timestamps) + 1,
                start=start,
                end=end,
                silence_removed=silence_signal_removed
            ))


            segmented_audio = np.concatenate((segmented_audio, audio_slice))

  
        
        return segmented_audio, vad_audio_timestamps          




