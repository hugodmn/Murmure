import whisper
import numpy as np 
import os 
from scipy.io.wavfile import write
from ..vad.types import VADSegmentOutput
from typing import List
from ..types import TranscriptionOutput
import torch




class TranscriptionModule():

    def __init__(self, model_size : str = "medium"):

        self.model_size = model_size
        self.model_dir_path = os.path.join(os.path.dirname(__file__), 'models')
        self.model_path = os.path.join(self.model_dir_path, model_size)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        os.makedirs("models", exist_ok = True)

        if not os.path.exists(self.model_path):
            print(f"Model '{model_size}' not found locally. Downloading...")
            self.model = whisper.load_model(model_size, download_root=self.model_path, device=self.device)
        else:
            print(f"Loading existing model '{model_size}' from '{self.model_path}'")
            self.model = whisper.load_model(model_size, download_root=self.model_path, device=self.device)


    def transcribe(self, 
                   audio_array : np.ndarray,
                   vad_audio_timestamps : List[VADSegmentOutput],
                   temperature : float = 0.0,
                   compression_ratio_threshold : int = 2,
                   condition_on_previous_text : bool = False,
                   beam_size : int = 5,
                   sample_rate : int = 16000
                   ) -> str :
        
        audio_array = audio_array.astype(np.float32)
       
        
        transcript = self.model.transcribe(audio_array,
                                           temperature = temperature,
                                           compression_ratio_threshold = compression_ratio_threshold,
                                           condition_on_previous_text = condition_on_previous_text,
                                           beam_size = beam_size,
                                           )
        
        transcript_segments = transcript['segments']

        transcript_segments = self.realign_whisper_segments(transcript_segments, 
                                                            vad_audio_timestamps = vad_audio_timestamps,
                                                            sample_rate=sample_rate)

        return transcript_segments
    

    def realign_whisper_segments(self, 
                                 transcript_segments : dict,
                                 vad_audio_timestamps : List[VADSegmentOutput],
                                 sample_rate : int = 16000) -> List[TranscriptionOutput] : 
        
        realigned_segments = []
        i = 1

        for segment in transcript_segments :

            for vad_seg_idx, vad_segment in enumerate(vad_audio_timestamps) :
                
                new_start = segment['start']
                new_end = segment['end']

                if vad_segment.start <= segment['start'] < vad_segment.end :

                    new_start += vad_segment.silence_removed

                    

                    if not vad_segment.start <= segment['end'] < vad_segment.end : 
                        new_end += vad_audio_timestamps[vad_seg_idx].silence_removed
                    
                    else : 
                        new_end += vad_segment.silence_removed

            realigned_segments.append(TranscriptionOutput(
                    idx = i,
                    transcription=segment['text'],
                    start = new_start,
                    end = new_end
                ))
            i+=1

        print(realigned_segments)

        return realigned_segments


