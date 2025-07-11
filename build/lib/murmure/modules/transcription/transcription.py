import os
import subprocess
import numpy as np
from typing import List
from ..types import TranscriptionOutput
from ..vad.types import VADSegmentOutput
from .whispercpp_utils import get_whispercpp_model
from utils.logger import get_logger
import re

logger = get_logger(__name__)

class TranscriptionModule:

    def __init__(self, 
                 model_size: str = "base",
                 device : str = 'cpu'):
        
        self.model_size = model_size
        self.model_dir_path = os.path.join(os.path.dirname(__file__), 'models')
        
        os.makedirs(self.model_dir_path, exist_ok=True)

        self.binary_path = os.path.join('/'.join(self.model_dir_path.split('/')[:-4]), "whisper.cpp/build/bin/whisper-cli")  # à adapter
        self.device = device

        if self.device == 'mps':
            self.model_path = get_whispercpp_model(self.model_size, self.model_dir_path)
        else:
            import whisper
            self.model_path = os.path.join(self.model_dir_path, self.model_size)
            if not os.path.exists(self.model_path):
                self.model = whisper.load_model(self.model_size, download_root=self.model_path)
            else:
                self.model = whisper.load_model(self.model_size, download_root=self.model_path)

        

    def transcribe(self, audio_array: np.ndarray, vad_audio_timestamps: List['VADSegmentOutput'],
                   temperature: float = 0.0, compression_ratio_threshold: int = 2,
                   condition_on_previous_text: bool = False, beam_size: int = 5,
                   sample_rate: int = 16000) -> List['TranscriptionOutput']:

        logger.info('--------- [STARTING] ---------')



        if self.device == 'mps':
                    
            import soundfile as sf
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                sf.write(tmp_audio.name, audio_array, samplerate=sample_rate)
                tmp_path = tmp_audio.name

            command = [
                self.binary_path,
                "-m", self.model_path,
                "-f", tmp_path,
                "-l", "auto",
                "--print-progress"
            ]

            logger.info(f"Running whisper.cpp")
            result = subprocess.run(command, capture_output=True, text=True)
            output = result.stdout.strip()
            os.remove(tmp_path)


            transcript_segments = []
            pattern = re.compile(r"\[(\d+:\d+:\d+\.\d+)\s+-->\s+(\d+:\d+:\d+\.\d+)\]\s+(.*)")

            for line in output.splitlines():
                match = pattern.match(line)
                if match:
                    start_str, end_str, text = match.groups()
                    start = _time_str_to_seconds(start_str)
                    end = _time_str_to_seconds(end_str)
                    transcript_segments.append({
                        "start": start,
                        "end": end,
                        "text": text.strip()
                    })

        else:

            logger.info('Running whisper')

            audio_array = audio_array.astype(np.float32)
            result = self.model.transcribe(audio_array,
                                           verbose=True,
                                           temperature=temperature,
                                           compression_ratio_threshold=compression_ratio_threshold,
                                           condition_on_previous_text=condition_on_previous_text,
                                           beam_size=beam_size)
            transcript_segments = result['segments']


        logger.info('--------- [FINISHING] ---------')


        return self.realign_whisper_segments(transcript_segments, vad_audio_timestamps, sample_rate)






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

        return realigned_segments
    


def _time_str_to_seconds(time_str: str) -> float:
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000