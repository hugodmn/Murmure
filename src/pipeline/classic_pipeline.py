from src.modules import TranscriptionModule, VADModule, SpeakerRecogntionModule
import numpy as np 
from src.utils import load_audio, write_transcription



class FullTranscriptPipeline():

    def __init__(self,
                 model_size : str,
                 do_vad : bool = True,
                 do_diarization : bool = True):

        self.do_vad = do_vad
        self.do_diarization = do_diarization

        self.vad_module = VADModule()
        self.transcription_module = TranscriptionModule(
            model_size=model_size
        )
        self.speaker_recognition = SpeakerRecogntionModule()

    
    def process(self,
                audio_path : str,
                  ):
        
        audio, sr = load_audio(audio_path = audio_path)

        if self.do_vad :
            speech_timestamps = self.vad_module.detect_speech_segments(audio = audio)

            segmented_audio, vad_audio_timestamps = self.vad_module.segmentation(audio = audio,
                                                           speech_timestamps = speech_timestamps)
            
        transcription_segments = self.transcription_module.transcribe(segmented_audio,
                                                                vad_audio_timestamps=vad_audio_timestamps
                                                                )

        if self.do_diarization : 
            transcription_segments = self.speaker_recognition.process_audio(transcription_segments=transcription_segments,
                                                                                    audio=audio)
        
        return transcription_segments
        

