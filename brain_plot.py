"""
brain_plot.py

Renders TRIBE v2 cortical activation maps onto brain surface images
using PlotBrainNilearn from tribev2.plotting.cortical.

Functions
---------
render_brain_image   — three-view (left, right, dorsal) activation PNG
render_network_highlight — same but annotates parcels for one network
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def render_brain_image(
    mean_activation: np.ndarray,
    output_path: str = "brain.png",
) -> str:
    """Render a three-view brain surface image and save it as a PNG.

    Parameters
    ----------
    mean_activation:
        1-D float array of shape (20484,) — mean activation per cortical
        vertex on the fsaverage5 mesh.
    output_path:
        Destination path for the saved PNG.

    Returns
    -------
    str — the resolved output path.
    """
    from tribev2.plotting.cortical import PlotBrainNilearn

    output_path = str(Path(output_path).resolve())
    views = ["left", "right", "dorsal"]

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    fig, axarr = plotter.get_fig_axes(views=views)

    plotter.plot_surf(
        signals=mean_activation,
        views=views,
        axes=list(axarr),
        cmap="hot",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Brain image saved to %s", output_path)
    return output_path


def render_network_highlight(
    mean_activation: np.ndarray,
    network_name: str,
    output_path: str = "brain_highlight.png",
) -> str:
    """Render a brain image with HCP ROI labels annotated for one network.

    Uses left and right lateral views only (annotate_rois does not support
    the dorsal / 'both' hemisphere view).

    Parameters
    ----------
    mean_activation:
        1-D float array of shape (20484,) — mean activation per vertex.
    network_name:
        Key from ATTENTION_NETWORKS (e.g. "dorsal_attention").
    output_path:
        Destination path for the saved PNG.

    Returns
    -------
    str — the resolved output path.
    """
    from attention_mapper import ATTENTION_NETWORKS
    from tribev2.plotting.cortical import PlotBrainNilearn

    output_path = str(Path(output_path).resolve())
    net_def = ATTENTION_NETWORKS.get(network_name)
    if net_def is None:
        raise ValueError(
            f"Unknown network '{network_name}'. "
            f"Valid keys: {list(ATTENTION_NETWORKS.keys())}"
        )

    parcel_names = net_def["parcels"]
    views = ["left", "right"]

    plotter = PlotBrainNilearn(mesh="fsaverage5")
    fig, axarr = plotter.get_fig_axes(views=views)

    plotter.plot_surf(
        signals=mean_activation,
        views=views,
        axes=list(axarr),
        cmap="hot",
        annotated_rois=parcel_names,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Network highlight image saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Quick self-test — run directly: python brain_plot.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile, os

    print("Running brain_plot self-test with fake activation data…")
    fake_activation = np.random.rand(20484).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = render_brain_image(fake_activation, output_path=os.path.join(tmpdir, "test_brain.png"))
        size = os.path.getsize(out)
        print(f"render_brain_image → {out}  ({size} bytes)")
        assert size > 0, "PNG file is empty!"

        out2 = render_network_highlight(
            fake_activation,
            network_name="dorsal_attention",
            output_path=os.path.join(tmpdir, "test_highlight.png"),
        )
        size2 = os.path.getsize(out2)
        print(f"render_network_highlight → {out2}  ({size2} bytes)")
        assert size2 > 0, "Highlight PNG file is empty!"

    print("brain_plot self-test PASSED.")
