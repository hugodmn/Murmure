import librosa 
from typing import Tuple, List 
import numpy as np 
from src.modules import TranscriptionOutput


def load_audio(audio_path : str,
               sample_rate : int = 16000) -> Tuple[np.ndarray, int]:

    audio, sr = librosa.load(audio_path, sr = sample_rate)

    return audio, sr



def write_transcription(audio_segments: List[TranscriptionOutput], 
                        file_path: str = "transcriptions/test.txt") -> None:
    """
    Writes the transcriptions from a list of AudioSegment objects to a text file.
    Merges consecutive segments if they are from the same speaker.
    
    :param audio_segments: List of AudioSegment objects containing start time, end time, transcription, and speaker ID.
    :param file_path: Path to the output .txt file.
    """

    merged_segments = []
    
    for segment in audio_segments:
        if merged_segments and merged_segments[-1]['speaker_id'] == segment.speaker_id:
            # Merge with the previous segment
            merged_segments[-1]['end'] = segment.end
            if segment.transcription[-1] in ['?', '.', '!']:
                merged_segments[-1]['transcription'] += "\n            " + segment.transcription
            else : 
                merged_segments[-1]['transcription'] +=  segment.transcription
        else:
            # Start a new segment
            merged_segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker_id': segment.speaker_id,
                'transcription': segment.transcription
            })

    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        for segment in merged_segments:
            f.write(f"({segment['start']:.2f} - {segment['end']:.2f} sec)\n")
            f.write(f"Speaker {segment['speaker_id']} : {segment['transcription']}\n\n")

    print(f"Transcription saved to {file_path}")

