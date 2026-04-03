"""
tribe_inference.py

Thin wrapper around TribeModel from tribev2.demo_utils.

Functions
---------
load_model        — loads TribeModel from HuggingFace Hub (cached)
run_inference     — runs model.predict() and returns mean vertex activation
download_youtube  — downloads a YouTube video to a local mp4 via yt-dlp
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# HuggingFace Hub model ID for TRIBE v2
_HF_REPO_ID = "facebook/tribev2"

# Extension → get_events_dataframe keyword argument
_INPUT_TYPE_KWARG = {
    "video": "video_path",
    "audio": "audio_path",
    "text": "text_path",
}

# File-extension → input_type
_SUFFIX_TO_TYPE: dict[str, str] = {
    ".mp4": "video",
    ".avi": "video",
    ".mkv": "video",
    ".mov": "video",
    ".webm": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".txt": "text",
}


def load_model(cache_folder: str = "./cache"):
    """Load TribeModel from the HuggingFace Hub (weights cached locally).

    Parameters
    ----------
    cache_folder:
        Directory used to cache extracted features and the downloaded
        model weights.

    Returns
    -------
    TribeModel instance ready for inference.
    """
    try:
        from tribev2.demo_utils import TribeModel
    except ImportError as exc:
        raise ImportError(
            "tribev2 is not installed. "
            "Run: pip install -e './tribev2[plotting]'"
        ) from exc

    logger.info("Loading TribeModel from '%s' …", _HF_REPO_ID)
    try:
        model = TribeModel.from_pretrained(
            _HF_REPO_ID,
            cache_folder=cache_folder,
        )
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "gated" in msg.lower() or "token" in msg.lower():
            raise RuntimeError(
                "HuggingFace authentication required. "
                "Accept the model licence at https://huggingface.co/facebook/tribev2 "
                "then run: huggingface-cli login"
            ) from exc
        raise

    logger.info("TribeModel loaded successfully.")
    return model


def run_inference(model, input_path: str, input_type: str) -> np.ndarray:
    """Run TRIBE v2 inference on a local file and return mean vertex activation.

    Parameters
    ----------
    model:
        TribeModel instance returned by load_model().
    input_path:
        Path to the local input file.
    input_type:
        One of "video", "audio", "text".

    Returns
    -------
    np.ndarray of shape (20484,) — mean activation per cortical vertex,
    averaged across all predicted time segments.
    """
    if input_type not in _INPUT_TYPE_KWARG:
        raise ValueError(
            f"input_type must be one of {list(_INPUT_TYPE_KWARG)}, "
            f"got '{input_type}'"
        )

    kwarg_name = _INPUT_TYPE_KWARG[input_type]
    logger.info("Building events DataFrame from %s (%s) …", input_path, input_type)
    events = model.get_events_dataframe(**{kwarg_name: input_path})

    logger.info("Running model.predict() …")
    preds, _segments = model.predict(events)
    # preds shape: (n_segments, 20484)
    mean_activation = preds.mean(axis=0)  # → (20484,)
    logger.info(
        "Inference complete: %d segments → mean activation shape %s",
        preds.shape[0],
        mean_activation.shape,
    )
    return mean_activation


def infer_input_type(file_path: str) -> str:
    """Infer input_type from file extension.

    Returns one of "video", "audio", "text".
    Raises ValueError if the extension is not recognised.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix not in _SUFFIX_TO_TYPE:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported: {list(_SUFFIX_TO_TYPE.keys())}"
        )
    return _SUFFIX_TO_TYPE[suffix]


def download_youtube(url: str, output_dir: str = "./tmp") -> str:
    """Download a YouTube video as mp4 using yt-dlp.

    Parameters
    ----------
    url:
        YouTube video URL.
    output_dir:
        Directory where the downloaded file is saved.

    Returns
    -------
    str — path to the downloaded mp4 file.

    Raises
    ------
    RuntimeError  if yt-dlp is not installed.
    subprocess.CalledProcessError  if the download fails.
    """
    if shutil.which("yt-dlp") is None:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp"
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Use a template so we know the exact filename afterwards
    output_template = str(output_dir_path / "%(id)s.%(ext)s")

    logger.info("Downloading YouTube video: %s", url)
    result = subprocess.run(
        [
            "yt-dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--output", output_template,
            "--no-playlist",
            "--print", "after_move:filepath",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    # yt-dlp prints the final filepath when --print after_move:filepath is used
    downloaded_path = result.stdout.strip().splitlines()[-1]
    logger.info("Downloaded to: %s", downloaded_path)
    return downloaded_path
