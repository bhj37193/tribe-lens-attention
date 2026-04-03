"""
app.py

Gradio web app for TRIBE Lens — Attention Analyzer.

Loads TribeModel once at startup. Accepts a YouTube URL or an uploaded
file, runs TRIBE v2 inference, maps activations onto attention networks,
renders a brain surface image, and returns a plain-English report.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import gradio as gr
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZeroGPU decorator (no-op when running locally without the spaces package)
# ---------------------------------------------------------------------------
try:
    import spaces

    GPU = spaces.GPU
except ImportError:
    def GPU(fn):
        return fn


# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
CACHE_FOLDER = os.environ.get("TRIBE_CACHE_FOLDER", "./cache")
_model = None


def _get_model():
    global _model
    if _model is None:
        from tribe_inference import load_model

        _model = load_model(cache_folder=CACHE_FOLDER)
    return _model


try:
    _model = _get_model()
except Exception as exc:
    logger.error("Model failed to load at startup: %s", exc)
    logger.error(
        "The app will still launch but inference will fail until the model "
        "is accessible. Accept the model licence at "
        "https://huggingface.co/facebook/tribev2 and run: huggingface-cli login"
    )

# ---------------------------------------------------------------------------
# Disclaimer text — always visible
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "⚠️ These are in-silico predictions based on population-average fMRI data. "
    "They estimate typical brain responses to this type of content on average — "
    "not a measurement of any individual's brain. "
    "This is not a clinical tool and makes no diagnostic claims."
)

# ---------------------------------------------------------------------------
# Core inference function (wrapped with @GPU for ZeroGPU)
# ---------------------------------------------------------------------------


@GPU
def _run_inference_gpu(model, input_path: str, input_type: str) -> np.ndarray:
    from tribe_inference import run_inference

    return run_inference(model, input_path, input_type)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


def analyze(youtube_url: str, uploaded_file):
    """Run the full pipeline and return (brain_image_path, summary, table).

    Parameters
    ----------
    youtube_url:
        YouTube video URL string (may be empty).
    uploaded_file:
        Gradio file object from gr.File — has a .name attribute pointing
        to the temp path on disk, or None.

    Returns
    -------
    tuple[str | None, str, list[dict]]
        Brain image path, attention summary text, results table rows.
    """
    from attention_mapper import (
        build_results_table,
        compute_activation_levels,
        generate_summary,
        get_network_activations,
    )
    from brain_plot import render_brain_image
    from tribe_inference import download_youtube, infer_input_type

    tmpdir = tempfile.mkdtemp(prefix="tribe_lens_")
    try:
        # ------------------------------------------------------------------
        # 1. Resolve input
        # ------------------------------------------------------------------
        youtube_url = (youtube_url or "").strip()
        has_url = bool(youtube_url)
        has_file = uploaded_file is not None

        if has_url and has_file:
            return (
                None,
                "Please provide either a YouTube URL or an uploaded file — not both.",
                [],
            )
        if not has_url and not has_file:
            return (
                None,
                "Please provide a YouTube URL or upload a video / audio / text file.",
                [],
            )

        if has_url:
            try:
                input_path = download_youtube(youtube_url, output_dir=tmpdir)
            except RuntimeError as exc:
                return None, f"Download error: {exc}", []
            except Exception as exc:
                return None, f"Failed to download video: {exc}", []
            input_type = "video"
        else:
            # Gradio passes the temp path as uploaded_file.name
            src_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file.name
            try:
                input_type = infer_input_type(src_path)
            except ValueError as exc:
                return None, str(exc), []
            # Copy to our temp dir so we can clean up predictably
            ext = Path(src_path).suffix
            input_path = str(Path(tmpdir) / f"input{ext}")
            shutil.copy2(src_path, input_path)

        # ------------------------------------------------------------------
        # 2. Inference
        # ------------------------------------------------------------------
        model = _get_model()
        if model is None:
            return (
                None,
                "Model is not loaded. Check the server logs for details.",
                [],
            )

        try:
            mean_activation = _run_inference_gpu(model, input_path, input_type)
        except Exception as exc:
            logger.exception("Inference failed")
            return None, f"Inference error: {exc}", []

        # ------------------------------------------------------------------
        # 3. Attention network mapping
        # ------------------------------------------------------------------
        try:
            network_scores = get_network_activations(mean_activation)
            levels = compute_activation_levels(network_scores)
            summary = generate_summary(network_scores, levels)
            table_rows = build_results_table(network_scores, levels)
        except Exception as exc:
            logger.exception("Attention mapping failed")
            return None, f"Analysis error: {exc}", []

        # ------------------------------------------------------------------
        # 4. Brain image rendering
        # ------------------------------------------------------------------
        brain_png = os.path.join(tmpdir, "brain.png")
        try:
            render_brain_image(mean_activation, output_path=brain_png)
        except Exception as exc:
            logger.exception("Brain rendering failed")
            brain_png = None
            summary += f"\n\n(Brain image could not be rendered: {exc})"

        # Convert table rows to the format Gradio Dataframe expects:
        # list of lists, matching the column order defined in the UI
        table_display = [
            [
                row["label"],
                f"{row['activation']:.4f}",
                row["level"],
                row["citation"],
            ]
            for row in table_rows
        ]

        # Copy brain PNG outside tmpdir so Gradio can serve it after cleanup
        if brain_png and os.path.exists(brain_png):
            persistent_png = tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix="tribe_brain_"
            )
            shutil.copy2(brain_png, persistent_png.name)
            brain_png_out = persistent_png.name
        else:
            brain_png_out = None

        return brain_png_out, summary, table_display

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="TRIBE Lens — Attention Analyzer") as demo:
    gr.Markdown(
        """
# TRIBE Lens — Attention Analyzer

**Predicts attention-relevant brain activation from video and audio content using
TRIBE v2 (Meta Research)**

Paste a YouTube link *or* upload a file. The model runs brain encoding inference
and maps predicted cortical activation onto named attention networks.
"""
    )

    gr.Markdown(f"> {_DISCLAIMER}")

    with gr.Row():
        youtube_input = gr.Textbox(
            label="YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            scale=3,
        )
        file_input = gr.File(
            label="Or upload a file",
            file_types=[".mp4", ".mp3", ".wav", ".txt"],
            scale=2,
        )

    analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        brain_img = gr.Image(
            label="Brain activation map",
            type="filepath",
        )

    summary_text = gr.Textbox(
        label="Attention summary",
        lines=4,
        interactive=False,
    )

    results_table = gr.Dataframe(
        headers=["Network", "Activation", "Level", "Citation"],
        label="Attention network breakdown",
        interactive=False,
        wrap=True,
    )

    gr.Markdown(
        f"""
---
**Disclaimer:** {_DISCLAIMER}

Built on [TRIBE v2](https://huggingface.co/facebook/tribev2) (Meta Research, d'Ascoli et al., 2026) ·
[HCP MMP1.0 parcellation](https://doi.org/10.1038/nature18933) (Glasser et al., 2016) ·
License: CC-BY-NC-4.0 (non-commercial use only)
"""
    )

    gr.Examples(
        examples=[["https://www.youtube.com/watch?v=dQw4w9WgXcQ", None]],
        inputs=[youtube_input, file_input],
        label="Example",
    )

    analyze_btn.click(
        fn=analyze,
        inputs=[youtube_input, file_input],
        outputs=[brain_img, summary_text, results_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
