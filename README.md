# TRIBE Lens — Attention Analyzer

A personal tool that predicts which attention-relevant brain regions are activated when you consume video or audio content, using Meta Research's TRIBE v2 brain encoding model.

Paste a YouTube link or upload a video. Get a rendered brain surface image and a plain-English breakdown of predicted cortical activation across attention networks — grounded in peer-reviewed neuroscience.

**Live demo:** [HuggingFace Space](https://huggingface.co/spaces/) *(link once deployed)*

---

## What this actually does

TRIBE v2 is a deep multimodal brain encoding model from Meta Research that predicts fMRI brain responses to naturalistic stimuli (video, audio, text). It was trained on real human brain recordings and maps multimodal inputs onto the cortical surface.

This tool takes that raw cortical prediction and translates it into named attention network activations using the HCP MMP1.0 parcellation — the same anatomical brain map used in human neuroscience research.

The output looks like this:

| Network | Activation | Level | Source |
|---------|-----------|-------|--------|
| Voluntary attention control | 0.74 | HIGH | Corbetta & Shulman, 2002 |
| Mind-wandering / attention withdrawal | 0.22 | LOW | Buckner et al., 2008 |
| Involuntary attention capture | 0.61 | MODERATE | Corbetta & Shulman, 2002 |
| Cognitive effort / working memory | 0.58 | MODERATE | Vincent et al., 2008 |
| Visual processing | 0.81 | VERY HIGH | Wandell et al., 2007 |
| Auditory / speech processing | 0.49 | MODERATE | Formisano et al., 2003 |
| Language comprehension | 0.55 | MODERATE | Fedorenko et al., 2011 |

Alongside a rendered brain surface image showing predicted activation across both hemispheres.

---

## Important disclaimer

These are **in-silico predictions** based on population-average fMRI data. They estimate how a typical brain responds to this type of content on average — not a measurement of your individual brain. This is not a clinical tool and makes no diagnostic claims. All outputs should be interpreted as research-grade signal, not personal neuroimaging.

---

## Attention networks explained

**Voluntary attention control (Dorsal Attention Network)**
Activates when you are actively tracking something. High activation means the content demanded sustained, directed focus. Source: Corbetta & Shulman, 2002, *Nature Reviews Neuroscience*.

**Mind-wandering / attention withdrawal (Default Mode Network)**
Activates when attention drifts away from the current stimulus. High default mode during content means the brain was disengaging. Source: Buckner et al., 2008, *Annals of the New York Academy of Sciences*.

**Involuntary attention capture (Ventral Attention Network)**
Activates when something grabs your attention automatically — not by choice. High activation suggests the content had high salience or surprise. Source: Corbetta & Shulman, 2002, *Nature Reviews Neuroscience*.

**Cognitive effort / working memory (Frontoparietal Control Network)**
Activates under high cognitive load — complex reasoning, task switching, holding information in mind. Source: Vincent et al., 2008, *Journal of Neurophysiology*.

**Visual processing**
Early and mid-level visual cortex activation. Driven by visual complexity, motion, and contrast. Source: Wandell et al., 2007, *Neuron*.

**Auditory / speech processing**
Primary auditory cortex and speech-selective regions. Driven by speech clarity, music, and audio complexity. Source: Formisano et al., 2003, *Neuron*.

**Language comprehension**
Left-lateralized language network. Activates during semantic processing and sentence comprehension. Source: Fedorenko et al., 2011, *PNAS*.

---

## Supported inputs

| Input type | Supported |
|-----------|-----------|
| YouTube link | Yes |
| Video file (.mp4, .avi, .mkv, .mov) | Yes |
| Audio file (.wav, .mp3, .flac) | Yes |
| Text file (.txt) | Yes |
| Instagram / TikTok links | No — platform restrictions |
| Images | No — TRIBE v2 has no image encoder |

---

## Setup — run locally on your machine

### Requirements

- Python 3.11 or higher
- Mac, Linux, or Windows
- At least 16GB RAM
- GPU optional but recommended (inference is 30 to 90 seconds on CPU, 10 to 20 seconds on GPU)

### 1. Clone this repo and the TRIBE v2 fork

```bash
git clone https://github.com/bhj37193/tribe-lens-attention
cd tribe-lens-attention

# clone TRIBE v2 into the same directory
git clone https://github.com/bhj37193/tribev2
cd tribev2
pip install -e ".[plotting]"
cd ..
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model weights

Weights download automatically from HuggingFace on first run. This is roughly 8 to 15GB so expect the first launch to take a few minutes depending on your connection. They are cached locally after that.

You need a free HuggingFace account and to accept the model terms at:
https://huggingface.co/facebook/tribev2

Then authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

### 4. Run the app

```bash
python app.py
```

Opens at `http://localhost:7860`

---

## Setup — deploy your own HuggingFace Space

### 1. Create a HuggingFace account and upgrade to Pro

Required for ZeroGPU access. Cost: $9/month.
https://huggingface.co/pricing

### 2. Create a new Space

- Go to huggingface.co/new-space
- Select Gradio as the SDK
- Select ZeroGPU as the hardware

### 3. Push this repo to your Space

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/tribe-lens-attention
git push space main
```

### 4. Accept TRIBE v2 model terms

Go to https://huggingface.co/facebook/tribev2 and accept the terms so your Space can download the weights.

### 5. Add your HuggingFace token as a Space secret

In your Space settings, add a secret named `HF_TOKEN` with your HuggingFace access token. This allows the Space to download the gated model weights.

The app will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/tribe-lens-attention`

---

## Cost to run

### Local (your own machine)

**$0/month.** You only need internet to download the weights the first time.

### HuggingFace Space (shared with others or accessible via URL)

| Item | Cost |
|------|------|
| HuggingFace Pro subscription | $9/month |
| ZeroGPU compute | Included in Pro (25 min/day H200) |
| Spaces hosting | Included in Pro |
| Model weights storage | Free (loaded from HuggingFace Hub) |
| yt-dlp (YouTube download) | Free |

**Total: $9/month** for personal use.

If you expect heavy usage from multiple users simultaneously, you would need a dedicated GPU instance starting at roughly $0.40/hour ($290/month always-on). For personal use ZeroGPU is sufficient.

---

## Project structure

```
tribe-lens-attention/
├── app.py                  # Gradio interface
├── tribe_inference.py      # TribeModel wrapper — loads model, runs predict()
├── attention_mapper.py     # Maps HCP parcels → attention networks with citations
├── brain_plot.py           # Renders brain surface image via PlotBrainNilearn
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How it works technically

1. Input (YouTube URL or file) is passed to `TribeModel.get_events_dataframe()`
2. TRIBE v2 converts the input to word-level events via audio extraction and transcription
3. `TribeModel.predict()` returns `preds` of shape `(n_segments, 20484)` — predicted fMRI activation across 20,484 cortical vertices on the fsaverage5 mesh, one row per ~2 second TR
4. Activations are averaged across segments to get a single `(20484,)` vector
5. `summarize_by_roi()` from `tribev2/utils.py` averages vertex values within each HCP MMP1.0 parcel
6. Parcel activations are grouped into attention networks using published functional network assignments
7. Activation levels are normalized relative to the distribution across all parcels and thresholded into LOW / MODERATE / HIGH / VERY HIGH
8. `PlotBrainNilearn` renders the activation map as a brain surface image
9. Results are returned to the Gradio interface

---

## Built on

- **TRIBE v2** — Meta Research. d'Ascoli et al., 2026. *A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience*. https://huggingface.co/facebook/tribev2

- **HCP MMP1.0 Parcellation** — Glasser et al., 2016. *A multi-modal parcellation of human cerebral cortex*. Nature, 536, 171–178.

- **Nilearn** — Abraham et al., 2014. *Machine learning for neuroimaging with scikit-learn*. Frontiers in Neuroinformatics.

---

## License

This project is licensed under **CC-BY-NC-4.0** (Creative Commons Attribution Non-Commercial 4.0 International), inherited from TRIBE v2.

**You are free to:**
- Use this code for personal, research, or educational purposes
- Share and adapt the code with attribution
- Deploy your own instance for non-commercial use

**You are not permitted to:**
- Use this code or any derivative for commercial purposes
- Sell access to this tool or incorporate it into a paid product
- Remove attribution to Meta Research (TRIBE v2) or this repository

Full license text: https://creativecommons.org/licenses/by-nc/4.0/

If you use this in research or a public project, please cite both TRIBE v2 and this repository.

```bibtex
@article{dAscoli2026TribeV2,
  title={A foundation model of vision, audition, and language for in-silico neuroscience},
  author={d'Ascoli, Stéphane and Rapin, Jérémy and Benchetrit, Yohann and Brookes, Teon
          and Begany, Katelyn and Raugel, Joséphine and Banville, Hubert and King, Jean-Rémi},
  year={2026}
}
```

---

## Contributing

Issues and pull requests welcome. This is a personal project built on top of Meta Research's TRIBE v2. Please keep contributions within the CC-BY-NC-4.0 license terms.

---

*Built by [@bhj37193](https://github.com/bhj37193)*
