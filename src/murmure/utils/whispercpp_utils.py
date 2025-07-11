import os
import requests
from tqdm import tqdm
import subprocess
from pathlib import Path
import stat

from murmure.utils.logger import get_logger

logger = get_logger(__name__)

WHISPER_CPP_REPO = "https://github.com/ggerganov/whisper.cpp"


def get_whispercpp_model(model_size: str, model_dir: str = "models") -> str:
    """
    Télécharge le modèle whisper.cpp si nécessaire, et retourne le chemin du fichier .bin
    """
    filename = f"ggml-{model_size}.bin"
    model_path = os.path.join(model_dir, filename)

    if not os.path.exists(model_path):
        logger.info(f"Model '{model_size}' not found. Downloading...")
        os.makedirs(model_dir, exist_ok=True)
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{filename}"
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Erreur {response.status_code} lors du téléchargement : {url}")

        total = int(response.headers.get("content-length", 0))
        with open(model_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {filename}"
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        print(f"Model found: {model_path}")

    return model_path


logger = get_logger("whispercpp_utils")


def ensure_whispercpp_installed(whispercpp_path: Path):
    binary_path = whispercpp_path / "build" / "bin" / "whisper-cli"

    if not binary_path.exists():
        raise FileNotFoundError(
            f"whisper.cpp binary not found at {binary_path}. "
            f"Please install whisper.cpp manually as explained in the README."
        )
    
    if not os.access(str(binary_path), os.X_OK):
        raise PermissionError(
            f"whisper.cpp binary at {binary_path} is not executable. "
            f"Try running: chmod +x {binary_path}"
        )

    logger.info("whisper.cpp is available.")
    return binary_path
