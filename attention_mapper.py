"""
attention_mapper.py

Maps TRIBE v2 cortical vertex activations onto named attention networks
using the HCP MMP1.0 parcellation.

Uses summarize_by_roi() and get_hcp_labels() from tribev2.utils — do not
reimplement those functions here.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# Seven attention-relevant networks from published literature.
# Parcel names match (or partially match) HCP MMP1.0 labels returned by
# get_hcp_labels(). Partial matching is used at runtime with warnings for
# unmatched parcels rather than crashing.
ATTENTION_NETWORKS = {
    "dorsal_attention": {
        "parcels": ["IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "FEF", "6r"],
        "label": "Voluntary attention control",
        "direction": "high_good",
        "citation": "Corbetta & Shulman, 2002, Nature Reviews Neuroscience",
    },
    "default_mode": {
        "parcels": ["PCC", "RSC", "31pd", "31pv", "d23ab", "v23ab", "7m"],
        "label": "Mind-wandering / attention withdrawal",
        "direction": "high_bad",
        "citation": "Buckner et al., 2008, Annals of the NY Academy of Sciences",
    },
    "ventral_attention": {
        "parcels": ["TPJ", "TPOJ1", "TPOJ2", "STV", "PFop", "PFt"],
        "label": "Involuntary attention capture",
        "direction": "neutral",
        "citation": "Corbetta & Shulman, 2002, Nature Reviews Neuroscience",
    },
    "frontoparietal": {
        "parcels": ["IFJa", "IFJp", "46", "9-46d", "IFSp", "IFSa"],
        "label": "Cognitive effort / working memory",
        "direction": "neutral",
        "citation": "Vincent et al., 2008, Journal of Neurophysiology",
    },
    "visual": {
        "parcels": ["V1", "V2", "V3", "V3A", "V4", "MT", "MST"],
        "label": "Visual processing",
        "direction": "neutral",
        "citation": "Wandell et al., 2007, Neuron",
    },
    "auditory": {
        "parcels": ["A1", "MBelt", "LBelt", "PBelt", "STSdp", "STSda"],
        "label": "Auditory / speech processing",
        "direction": "neutral",
        "citation": "Formisano et al., 2003, Neuron",
    },
    "language": {
        "parcels": ["IFGop", "IFGs", "44", "45", "47l", "TE1a", "TE1p"],
        "label": "Language comprehension",
        "direction": "neutral",
        "citation": "Fedorenko et al., 2011, PNAS",
    },
}


def _build_parcel_activation_dict(
    mean_activation: np.ndarray, mesh: str = "fsaverage5"
) -> dict[str, float]:
    """Call summarize_by_roi and pair results with parcel names.

    Returns a dict mapping each HCP parcel name to its mean activation.
    """
    from tribev2.utils import get_hcp_labels, summarize_by_roi

    roi_values = summarize_by_roi(mean_activation, hemi="both", mesh=mesh)
    labels = list(get_hcp_labels(mesh=mesh, combine=False, hemi="both").keys())
    assert len(labels) == len(roi_values), (
        f"Label count {len(labels)} != roi_values count {len(roi_values)}"
    )
    return dict(zip(labels, roi_values.tolist()))


def _match_parcels(
    wanted: list[str], available: dict[str, float]
) -> dict[str, float]:
    """Find activation values for each wanted parcel using partial matching.

    For each name in `wanted`, look for an exact match first, then try
    case-insensitive prefix matching.  Logs a warning for any unmatched
    parcel but does not raise.

    Returns a dict mapping matched parcel names to their activation values.
    """
    matched: dict[str, float] = {}
    available_lower = {k.lower(): (k, v) for k, v in available.items()}

    for parcel in wanted:
        lower = parcel.lower()
        if parcel in available:
            matched[parcel] = available[parcel]
        elif lower in available_lower:
            real_name, val = available_lower[lower]
            matched[real_name] = val
        else:
            # partial prefix match
            candidates = [
                (k, v)
                for k, v in available.items()
                if k.lower().startswith(lower) or lower.startswith(k.lower())
            ]
            if candidates:
                name, val = candidates[0]
                matched[name] = val
                if len(candidates) > 1:
                    logger.warning(
                        "Parcel '%s' matched multiple candidates %s; using '%s'",
                        parcel,
                        [c[0] for c in candidates],
                        name,
                    )
                else:
                    logger.debug("Parcel '%s' matched as '%s'", parcel, name)
            else:
                logger.warning(
                    "Parcel '%s' not found in HCP labels — skipping", parcel
                )

    return matched


def get_network_activations(
    mean_activation: np.ndarray, mesh: str = "fsaverage5"
) -> dict[str, float]:
    """Map vertex-level activations onto the seven attention networks.

    Parameters
    ----------
    mean_activation:
        1-D array of shape (20484,) — mean activation per cortical vertex
        on the fsaverage5 mesh, as returned by averaging TRIBE v2 preds
        across time segments.
    mesh:
        fsaverage mesh resolution.  Defaults to "fsaverage5".

    Returns
    -------
    dict mapping network key → mean activation score (float).
    Networks with no matched parcels are omitted with a warning.
    """
    parcel_activations = _build_parcel_activation_dict(mean_activation, mesh=mesh)

    network_scores: dict[str, float] = {}
    for network_key, network_def in ATTENTION_NETWORKS.items():
        matched = _match_parcels(network_def["parcels"], parcel_activations)
        if not matched:
            logger.warning(
                "Network '%s' has no matched parcels — omitted from results",
                network_key,
            )
            continue
        network_scores[network_key] = float(np.mean(list(matched.values())))
        logger.debug(
            "Network '%s': %.4f (from %d/%d parcels)",
            network_key,
            network_scores[network_key],
            len(matched),
            len(network_def["parcels"]),
        )

    return network_scores


def compute_activation_levels(network_scores: dict[str, float]) -> dict[str, str]:
    """Convert raw network scores to categorical activation levels.

    Thresholds are computed from the distribution across all networks in
    the current sample (not global population statistics) so the labels
    are relative comparisons, not absolute ones.

    Levels:
      LOW       — below 25th percentile
      MODERATE  — 25th to 75th percentile
      HIGH      — 75th to 90th percentile
      VERY HIGH — above 90th percentile
    """
    if not network_scores:
        return {}

    values = np.array(list(network_scores.values()), dtype=float)
    p25, p75, p90 = np.percentile(values, [25, 75, 90])

    levels: dict[str, str] = {}
    for network, score in network_scores.items():
        if score >= p90:
            levels[network] = "VERY HIGH"
        elif score >= p75:
            levels[network] = "HIGH"
        elif score >= p25:
            levels[network] = "MODERATE"
        else:
            levels[network] = "LOW"

    return levels


def generate_summary(
    network_scores: dict[str, float], levels: dict[str, str]
) -> str:
    """Generate a 2–3 sentence plain-English description of the activation pattern.

    Applies interpretation logic from the PRD.  No clinical claims; always
    includes the population-average caveat.
    """
    sentences: list[str] = []

    dorsal_level = levels.get("dorsal_attention", "MODERATE")
    dmn_level = levels.get("default_mode", "MODERATE")
    ventral_level = levels.get("ventral_attention", "MODERATE")
    frontal_level = levels.get("frontoparietal", "MODERATE")

    high_set = {"HIGH", "VERY HIGH"}

    if dorsal_level in high_set and dmn_level not in high_set:
        sentences.append(
            "Content engaged sustained voluntary attention — the brain was"
            " actively tracking the stimulus."
        )
    elif dmn_level in high_set and dorsal_level not in high_set:
        sentences.append(
            "Attention drifted during this content. Default mode dominance"
            " suggests the brain was mind-wandering rather than tracking the"
            " stimulus closely."
        )
    elif dorsal_level in high_set and dmn_level in high_set:
        sentences.append(
            "Both sustained attention and default mode circuits showed elevated"
            " activity, suggesting fluctuating engagement with the content."
        )
    else:
        sentences.append(
            "Attention-related circuits showed moderate, balanced activation"
            " across networks."
        )

    if ventral_level in high_set:
        sentences.append(
            "Content also captured attention involuntarily — the stimulus had"
            " high salience or contained surprising elements."
        )

    if frontal_level in high_set:
        sentences.append(
            "High cognitive load was predicted: frontoparietal circuits"
            " associated with working memory and task control were strongly"
            " activated."
        )

    sentences.append(
        "Note: these are in-silico predictions based on population-average"
        " fMRI data, not a measurement of any individual's brain."
    )

    return " ".join(sentences)


def build_results_table(
    network_scores: dict[str, float], levels: dict[str, str]
) -> list[dict]:
    """Return a sorted list of dicts for display in the Gradio results table.

    Keys: network, label, activation, level, citation.
    Sorted by activation score descending.
    """
    rows: list[dict] = []
    for network_key, score in network_scores.items():
        net_def = ATTENTION_NETWORKS[network_key]
        rows.append(
            {
                "network": network_key,
                "label": net_def["label"],
                "activation": round(score, 4),
                "level": levels.get(network_key, "MODERATE"),
                "citation": net_def["citation"],
            }
        )
    rows.sort(key=lambda r: r["activation"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Quick self-test — run directly: python attention_mapper.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running attention_mapper self-test with fake activation data…")
    fake_activation = np.random.rand(20484).astype(np.float32)

    scores = get_network_activations(fake_activation)
    print(f"\nNetwork scores: {scores}")

    lvls = compute_activation_levels(scores)
    print(f"\nActivation levels: {lvls}")

    summary = generate_summary(scores, lvls)
    print(f"\nSummary:\n{summary}")

    table = build_results_table(scores, lvls)
    print("\nResults table:")
    for row in table:
        print(f"  {row['label']:40s} {row['activation']:.4f}  {row['level']:10s}  {row['citation']}")
