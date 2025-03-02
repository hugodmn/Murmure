from modules import TranscriptionModule, VADModule, SpeakerRecogntionModule
import numpy as np 
from utils import load_audio, write_transcription



class FullTranscriptPipeline():

    def __init__(self,):

        self.vad_module = VADModule()
        self.transcription_module = TranscriptionModule()
        self.speaker_recognition = SpeakerRecogntionModule()

    
    def process(self,
                audio_path : str,
                do_vad_bool : bool = True,
                do_diarization : bool = True,
                  ):
        
        audio, sr = load_audio(audio_path = audio_path)

        if do_vad_bool :
            speech_timestamps = self.vad_module.detect_speech_segments(audio = audio)

            segmented_audio, vad_audio_timestamps = self.vad_module.segmentation(audio = audio,
                                                           speech_timestamps = speech_timestamps)
            
        transcription_segments = self.transcription_module.transcribe(segmented_audio,
                                                                vad_audio_timestamps=vad_audio_timestamps)

        if do_diarization : 
            transcription_segments = self.speaker_recognition.process_audio(transcription_segments=transcription_segments,
                                                                                    audio=audio)
        
        write_transcription(transcription_segments)
        



if __name__ == '__main__':
    Pipe = FullTranscriptPipeline()
    Pipe.process('test.wav')