import argparse
import os
import glob
from src.pipeline.classic_pipeline import FullTranscriptPipeline
from src.utils import write_transcription  # Adjust import if needed

def process_audio(path, model_size, use_vad, use_diarization):
    """
    Process an audio file or all audio files in a folder using FullTranscriptPipeline.
    """
    Pipe = FullTranscriptPipeline(model_size=model_size, vad=use_vad, diarization=use_diarization)

    if os.path.isfile(path):  
        print(f"Processing single file: {path}")
        transcription_segments = Pipe.process(path)
        write_transcription(transcription_segments)

    elif os.path.isdir(path):  
        audio_files = glob.glob(os.path.join(path, "*.wav")) + glob.glob(os.path.join(path, "*.mp3"))

        if not audio_files:
            print("No audio files found in the folder.")
            return
        
        print(f"Processing {len(audio_files)} audio files in: {path}")
        for file in audio_files:
            print(f"Processing {file}...")
            transcription_segments = Pipe.process(file)
            write_transcription(transcription_segments)

    else:
        print("Invalid path provided. Please specify a valid file or directory.")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper, VAD, and Speaker Diarization.")
    
    parser.add_argument("path", type=str, help="Path to an audio file or a directory containing audio files (.wav, .mp3).")
    parser.add_argument("--model", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium).")
    parser.add_argument("--vad", action="store_true", help="Enable Voice Activity Detection (VAD).", default = True)
    parser.add_argument("--diarization", action="store_true", help="Enable Speaker Diarization.", default = True)

    args = parser.parse_args()

    # Process file/folder with given arguments
    process_audio(args.path, args.model, args.vad, args.diarization)

if __name__ == "__main__":
    main()
