from modules import TranscriptionModule, VADModule, SpeakerRecogntionModule
import numpy as np 
from utils import load_audio



class FullTranscriptPipeline():

    def __init__(self,):

        self.vad_module = VADModule()
        self.transcription_module = TranscriptionModule()
        self.speaker_recognition = SpeakerRecogntionModule()

    
    def process(self,
                audio_path : str,
                do_vad_bool : bool = True
                  ):
        
        audio, sr = load_audio(audio_path = audio_path)

        if do_vad_bool :
            speech_timestamps = self.vad_module.detect_speech_segments(audio = audio)

            segmented_audio, vad_segmented_audio_info = self.vad_module.segmentation(audio = audio,
                                                           speech_timestamps = speech_timestamps)
            
        whisper_segments = self.transcription_module.transcribe(segmented_audio,
                                                                vad_segmented_audio_info=vad_segmented_audio_info)

        speaker_recognition_segments = self.speaker_recognition.process_audio(whisper_segments=whisper_segments,
                                                                                   audio=audio)
        # print(transcription['segments'])
        print(speaker_recognition_segments)



if __name__ == '__main__':
    Pipe = FullTranscriptPipeline()
    Pipe.process('test.wav')