from pathlib import Path
from .utils import choose_device


class Settings:
    CURRENT_FILE = Path(__file__).resolve()
    REPO_ROOT = CURRENT_FILE.parents[3]

    MODELS_DIR = REPO_ROOT / "models"
    WHISPER_CPP_CLI_PATH = REPO_ROOT / "whisper.cpp" / "build" / "bin" / "whisper-cli"

    MODEL_SIZE = "medium"
    ENABLE_VAD = True
    ENABLE_DIARIZATION = True
    DEVICE = choose_device()

    @classmethod
    def update_from_args(cls, args):
        """Override settings with CLI arguments."""
        cls.MODEL_SIZE = args.model
        cls.ENABLE_VAD = args.vad
        cls.ENABLE_DIARIZATION = args.diarization

        if hasattr(args, "device") and args.device is not None:
            cls.DEVICE = args.device
