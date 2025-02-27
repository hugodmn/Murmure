from modules import TranscriptionModule, VADModule
import numpy as np 
from utils import load_audio



class FullTranscriptPipeline():

    def __init__(self,):

        self.vad_module = VADModule()
        self.transcription_module = TranscriptionModule()

    
    def process(self,
                audio_path : str,
                do_vad_bool : bool = True
                  ):
        
        audio, sr = load_audio(audio_path = audio_path)

        if do_vad_bool :
            speech_timestamps = self.vad_module.detect_speech_segments(audio = audio)

            segmented_audio = self.vad_module.segmentation(audio = audio,
                                                           speech_timestamps = speech_timestamps)
            
        transcription = self.transcription_module.transcribe(segmented_audio)

        print(transcription)



if __name__ == '__main__':
    Pipe = FullTranscriptPipeline()
    Pipe.process('test.wav')