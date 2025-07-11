import numpy as np 

from murmure.modules import TranscriptionModule, VADModule, SpeakerDiarizationModule
from murmure.utils.utils import load_audio, remove_small_segments
from murmure.utils.settings import Settings

from murmure.utils.logger import get_logger

logger = get_logger(__name__)

class FullTranscriptPipeline():

    def __init__(self,
                 settings : Settings):

        self.device = settings.DEVICE
        self.do_vad = settings.ENABLE_VAD
        self.do_diarization = settings.ENABLE_DIARIZATION

        self.vad_module = VADModule(
            model_dir_path=settings.MODELS_DIR
        )

        self.transcription_module = TranscriptionModule(
            model_dir_path=settings.MODELS_DIR,
            whispercpp_path=settings.WHISPER_CPP_CLI_PATH,
            model_size=settings.MODEL_SIZE,
            device=settings.DEVICE
        )

        self.speaker_recognition = SpeakerDiarizationModule(
            model_dir_path=settings.MODELS_DIR
        )

    
    def process(self,
                audio_path : str,
                  ):
        
        audio, sr = load_audio(audio_path = audio_path)


        if self.do_vad :
            speech_timestamps = self.vad_module.detect_speech_segments(audio = audio)

            segmented_audio, vad_audio_timestamps = self.vad_module.segmentation(audio = audio,
                                                           speech_timestamps = speech_timestamps)
            
        transcription_segments = self.transcription_module.transcribe(segmented_audio,
                                                                vad_audio_timestamps=vad_audio_timestamps)
        
        transcription_segments = remove_small_segments(transcription_segments)

        if self.do_diarization and self.device != 'mps': 
            transcription_segments = self.speaker_recognition.process_audio(transcription_segments=transcription_segments,
                                                                                    audio=audio)
        elif self.do_diarization and self.device == 'mps':
            logger.info('Speaker Diarization is not available on mac')

        return transcription_segments
        



