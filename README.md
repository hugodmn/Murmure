# Audio Transcription Pipeline

## Overview
This project provides an **automatic speech transcription** pipeline using:
- **Whisper** for speech-to-text transcription.
- **Silero VAD** for **Voice Activity Detection (VAD)**.
- **ECAPA-TDNN** embeddings with **Agglomerative Clustering** for **Speaker Diarization**.

The pipeline can process:
- **Single audio files** (`.wav`, `.mp3`).
- **Folders of audio files** (batch processing).
- **Different Whisper model sizes** (`tiny`, `base`, `small`, `medium`, `large`).
- **Optional VAD and Speaker Diarization**.

## Installation

### Clone the repository

```sh
git clone https://github.com/hugodmn/Murmure.git
cd Murmure
```

### Install dependencies

```sh
pip install -r requirements.txt
```

## Usage

Run the `main.py` script with different options.

### Transcribe a Single Audio File

```sh
python main.py path/to/audio.wav
```

### Process All Files in a Folder

```sh
python main.py path/to/folder
```

### Change the Whisper Model Size

```sh
python main.py path/to/audio.wav --model large
```
**Available model sizes:**
- **`tiny`**
- **`base`**
- **`small`**
- **`medium`** (default)
- **`large`**

## Disable Voice Activity Detection (VAD)

By default, **Voice Activity Detection (VAD)** using **Silero VAD** is enabled.  
If you want to **disable VAD**, use:

```sh
python main.py path/to/audio.wav --no-vad
```

### Disable Speaker Diarization 

By default, Speaker Diarization using ECAPA-TDNN is enabled.
If you want to disable diarization, use:

```sh
python main.py path/to/audio.wav --diarization False 
```

