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
                   vad_segmented_audio_info : dict,
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
                                                            vad_segmented_audio_info = vad_segmented_audio_info,
                                                            sample_rate=sample_rate)

        return transcript_segments
    

    def realign_whisper_segments(self, 
                                 transcript_segments : dict,
                                 vad_segmented_audio_info : dict,
                                 sample_rate : int = 16000) -> dict : 
        
        for segment in transcript_segments :
            for vad_segment_info in vad_segmented_audio_info :
                if vad_segment_info['start'] <= segment['start']*sample_rate < vad_segment_info['end'] :
                    segment['start'] = segment['start'] + int(vad_segment_info['silence_signal_removed']/sample_rate)

                if vad_segment_info['start'] <= segment['end']*sample_rate < vad_segment_info['end'] :
                    segment['end'] = segment['end'] + int(vad_segment_info['silence_signal_removed']/sample_rate)

        return transcript_segments


