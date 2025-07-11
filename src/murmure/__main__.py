import argparse
import os
import glob
from murmure.pipeline.classic_pipeline import FullTranscriptPipeline
from murmure.utils.utils import write_transcription  
from murmure.utils.logger import get_logger
from murmure.utils.settings import Settings

logger = get_logger(__name__)

def process_audio(path : str, 
                  settings : Settings):
    """
    Process an audio file or all audio files in a folder using FullTranscriptPipeline.
    """

    Pipe = FullTranscriptPipeline(settings=settings)

    if os.path.isfile(path):  
        logger.info(f"Processing single file: {path}")
        transcription_segments = Pipe.process(path)
        write_transcription(transcription_segments,
                            file_name=path.split('/')[-1].split('.')[0])

    elif os.path.isdir(path):  
        audio_files = glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.mp3"))

        if not audio_files:
            logger.error("No audio files found in the folder.")
            return
        
        print(f"Processing {len(audio_files)} audio files in: {path}")
        for file in audio_files:
            print(f"Processing {file}...")
            transcription_segments = Pipe.process(file)
            write_transcription(transcription_segments,
                                file_name=file.split('/')[-1].split('.')[0])

    else:
        logger.error("Invalid path provided. Please specify a valid file or directory.")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper, VAD, and Speaker Diarization.")
    
    parser.add_argument("path", type=str, help="Path to an audio file or a directory containing audio files (.wav, .mp3).")
    parser.add_argument("--model", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium).")
    parser.add_argument("--no-vad", action="store_false", dest="vad", help="Disable Voice Activity Detection (VAD).")
    parser.add_argument("--no-diarization", action="store_false", dest="diarization", help="Disable Speaker Diarization.")

    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], help="Device to use (default: auto-detect)")

    settings = Settings()
    args = parser.parse_args()
    
    settings.update_from_args(args)
    
    # Process file/folder with given arguments
    process_audio(args.path, settings)

if __name__ == "__main__":
    main()
