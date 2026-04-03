# TRIBE Lens — Attention Analyzer
## Product Requirements Document

---

## What this is

A personal web app deployed on HuggingFace Spaces that takes a YouTube link or uploaded
video/audio file, runs it through TRIBE v2, and returns a plain-English report of which
attention-relevant brain regions were most activated, with a rendered brain surface image
and one-line research-backed explanations per region.

The output framing is:
  "Dorsal attention network: HIGH — region associated with top-down voluntary attention
   control (Corbetta & Shulman, 2002)"

Not emotion percentages. Not diagnostic claims. Grounded region labels with activation
levels and a single credible citation per region.

---

## Use case

Bo pastes a YouTube link or uploads a video. The app runs TRIBE v2 inference, maps the
cortical output onto named HCP parcellation regions, filters to attention-relevant
networks, and returns:

1. A rendered brain surface image (matplotlib, saved as PNG) showing activation
2. A ranked list of top activated attention regions with activation level and explanation
3. A plain-English summary: "This content activated sustained attention circuits more
   than default mode — consistent with content that demands continuous tracking."

---

## What is hard to vary (grounded constraints)

### TRIBE v2 output
- Shape: (n_segments, 20484 vertices) on fsaverage5
- Left hemisphere: vertices 0 to 10241
- Right hemisphere: vertices 10242 to 20483
- Each segment = one TR (~2 seconds of brain time, offset 5s for hemodynamic lag)
- Population average subject (not individual)

### HCP parcellation (already in codebase)
- `get_hcp_labels(mesh="fsaverage5")` returns named HCP MMP1.0 parcels → vertex indices
- `summarize_by_roi(data)` averages vertex activations per parcel
- `get_topk_rois(data, k=10)` returns the 10 highest activated parcel names
- These are REAL anatomical labels, not invented categories

### Attention networks from published literature
The following HCP parcel groups map to attention networks.
These are fixed by neuroscience, not by interpretation:

DORSAL ATTENTION NETWORK (Corbetta & Shulman, 2002, Nature Reviews Neuroscience)
- Parcels: IPS1, IPS2, IPS3, IPS4, IPS5, FEF, PFCL, 6r
- Function: top-down voluntary attention, spatial orienting
- Activation label: "Voluntary attention control"

DEFAULT MODE NETWORK (Buckner et al., 2008, Annals of the New York Academy of Sciences)
- Parcels: PCC, RSC, 31pd, 31pv, d23ab, v23ab, 7m, VMV1, VMV2, VMV3
- Function: mind-wandering, self-referential thought, attention AWAY from stimulus
- Activation label: "Mind-wandering / attention withdrawal"
- Note: HIGH default mode = attention is NOT on the content

VENTRAL ATTENTION / SALIENCE NETWORK (Corbetta & Shulman, 2002)
- Parcels: TPJ, TPOJ1, TPOJ2, TPOJ3, STV, PFop, PFt
- Function: stimulus-driven attention capture, reorienting
- Activation label: "Involuntary attention capture"

FRONTOPARIETAL CONTROL NETWORK (Vincent et al., 2008, Journal of Neurophysiology)
- Parcels: IFJa, IFJp, IFSp, IFSa, 46, 9-46d, 9-46v, p9-46v
- Function: cognitive control, working memory, task switching
- Activation label: "Cognitive effort / working memory load"

VISUAL CORTEX (Wandell et al., 2007, Neuron)
- Parcels: V1, V2, V3, V3A, V3B, V4, V6, V6A, MT, MST
- Function: early and mid-level visual processing
- Activation label: "Visual processing"

AUDITORY CORTEX (Formisano et al., 2003, Neuron)
- Parcels: A1, MBelt, LBelt, PBelt, RI, STSdp, STSda, STSvp, STSva
- Function: auditory processing, speech perception
- Activation label: "Auditory / speech processing"

LANGUAGE NETWORK (Fedorenko et al., 2011, PNAS)
- Parcels: IFGop, IFGs, 44, 45, 47l, TE1a, TE1p, TE2a, TE2p, PHT
- Function: language comprehension, semantic processing
- Activation label: "Language comprehension"

---

## Attention interpretation logic

After getting ROI activations from TRIBE, compute these composite scores:

```python
ATTENTION_NETWORKS = {
    "dorsal_attention": {
        "parcels": ["IPS1", "IPS2", "IPS3", "IPS4", "IPS5", "FEF", "6r"],
        "label": "Voluntary attention control",
        "direction": "high_good",  # high = more focused attention
        "citation": "Corbetta & Shulman, 2002, Nature Reviews Neuroscience",
    },
    "default_mode": {
        "parcels": ["PCC", "RSC", "31pd", "31pv", "d23ab", "v23ab", "7m"],
        "label": "Mind-wandering / attention withdrawal",
        "direction": "high_bad",  # high = attention drifting away
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
```

Activation level thresholds (relative to mean across all networks):
- LOW: below 25th percentile
- MODERATE: 25th to 75th percentile
- HIGH: above 75th percentile
- VERY HIGH: above 90th percentile

Plain-English summary logic:
- If dorsal_attention HIGH and default_mode LOW → "Content engaged sustained voluntary
  attention. The brain was tracking the stimulus actively."
- If default_mode HIGH and dorsal_attention LOW → "Attention drifted during this content.
  Default mode dominance suggests mind-wandering."
- If ventral_attention HIGH → "Content captured attention involuntarily — the stimulus
  had high salience."
- If frontoparietal HIGH → "High cognitive load. The brain was working hard to process
  this content."

---

## Input handling

YouTube links:
- Use yt-dlp to download video as mp4
- Pass to TribeModel.get_events_dataframe(video_path=...)
- yt-dlp is free, no API key needed for personal use

Uploaded files:
- Accept .mp4, .mp3, .wav, .txt
- Pass directly to get_events_dataframe with appropriate path argument

Text input (stretch goal):
- Write to temp .txt file
- Goes through gTTS → transcription → word events internally

---

## Output format

### Brain image
- Use PlotBrainNilearn from tribev2.plotting.cortical
- Views: ["left", "right", "dorsal"] — three angles
- Color map: "hot" (standard for activation)
- Save as PNG, display in Gradio

### Region table
For each attention network, one row:
| Network | Activation | Level | Citation |
|---------|-----------|-------|----------|
| Voluntary attention control | 0.73 | HIGH | Corbetta & Shulman, 2002 |
| Mind-wandering | 0.21 | LOW | Buckner et al., 2008 |

### Plain-English summary
2 to 3 sentences. No clinical claims. No dramatic language.
Describes what the activation pattern means for attention in plain terms.

---

## Tech stack

```
gradio                    # UI and HuggingFace Spaces hosting
tribev2                   # brain encoding model (local install from forked repo)
yt-dlp                    # YouTube video download
nilearn                   # brain surface plotting (already a dependency)
matplotlib                # figure rendering
numpy                     # array ops
torch                     # inference
mne                       # HCP parcellation loading (already in tribev2/utils.py)
tempfile / pathlib        # temp file management
```

---

## File structure

```
tribe-lens-attention/
├── app.py                    # main Gradio app
├── tribe_inference.py        # TribeModel loading and predict() wrapper
├── attention_mapper.py       # ROI → attention network mapping
├── brain_plot.py             # brain surface rendering → PNG
├── requirements.txt
└── README.md
```

---

## app.py structure (Gradio)

```python
import gradio as gr
from tribe_inference import load_model, run_inference
from attention_mapper import map_to_attention_networks, generate_summary
from brain_plot import render_brain

model = load_model()  # loads once at startup

def analyze(youtube_url, uploaded_file):
    # 1. get input
    # 2. run inference → preds (n_segments, 20484), segments
    # 3. mean over segments → (20484,) mean activation per vertex
    # 4. map to attention networks
    # 5. render brain image
    # 6. generate summary
    # 7. return image, table, summary text

with gr.Blocks() as demo:
    gr.Markdown("## TRIBE Lens — Attention Analyzer")
    with gr.Row():
        youtube_input = gr.Textbox(label="YouTube URL")
        file_input = gr.File(label="Or upload video / audio")
    analyze_btn = gr.Button("Analyze")
    brain_img = gr.Image(label="Brain activation")
    summary_text = gr.Textbox(label="Summary")
    results_table = gr.Dataframe(label="Region breakdown")
    
    analyze_btn.click(
        fn=analyze,
        inputs=[youtube_input, file_input],
        outputs=[brain_img, summary_text, results_table]
    )

demo.launch()
```

---

## HuggingFace Spaces deployment

1. Create new Space → Gradio SDK
2. Enable ZeroGPU in hardware settings (requires Pro, $9/month)
3. Add @spaces.GPU decorator to the inference function
4. Model weights load from "facebook/tribev2" on HuggingFace Hub automatically
5. Push code via git

requirements.txt includes:
- tribev2 installed from forked GitHub repo
- yt-dlp
- gradio
- spaces (ZeroGPU decorator)

---

## What this is NOT

- Not a clinical tool
- Not a real-time individual brain scanner
- Not a diagnostic instrument
- Outputs are population-average predictions, not measurements of Bo's actual brain
- All outputs carry this disclaimer: "These are in-silico predictions based on
  population-average fMRI data. They estimate typical brain responses to this type
  of content, not individual measurements."

---

## License note

TRIBE v2 is CC-BY-NC-4.0. This app is for personal, non-commercial use only.
The forked repo (bhj37193/tribev2) is the code source.

---

## Build order for VS Code

1. attention_mapper.py — pure Python, no ML, testable immediately
2. brain_plot.py — test with fake activation data
3. tribe_inference.py — requires model download, test last
4. app.py — wire everything together
5. Deploy to HuggingFace Spaces
