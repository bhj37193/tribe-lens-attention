---
title: TRIBE Lens Attention Analyzer
emoji: 🧠
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
---

# TRIBE Lens: Attention Analyzer

TRIBE Lens takes one input that TRIBE v2 can encode, runs Meta Research's TRIBE v2 model on it, averages the predicted cortical response over time, maps those vertex level predictions onto HCP MMP1.0 parcels, groups selected parcels into named attention related networks, and returns two outputs:

1. a rendered brain surface image
2. a network table with activation values and source citations

Accepted inputs:
- YouTube URL
- video file
- audio file
- plain text file

Rejected inputs:
- Instagram and TikTok links
- images

This is a research interface for model output inspection. It is not a tool for measuring a specific person's brain state.

---

## What the tool does

TRIBE v2 is a multimodal brain encoding model from Meta Research. Given naturalistic stimuli such as video, audio, or text, it predicts fMRI like cortical responses on the fsaverage5 surface.

This app does not claim to detect attention directly from a webcam, eye tracking stream, or EEG signal. It does one narrower thing.

It takes TRIBE v2's predicted cortical activity and converts that activity into a readable summary across seven named systems:

- Voluntary attention control
- Mind wandering and attention withdrawal
- Involuntary attention capture
- Cognitive effort and working memory
- Visual processing
- Auditory and speech processing
- Language comprehension

Each system score comes from parcel level aggregation, not from a classifier trained to output labels such as "engaged" or "bored."

---

## What the output means

For each run, the app returns a table with four fields:

| Network | Activation | Level | Source |
|---------|-----------|-------|--------|
| Voluntary attention control | 0.74 | HIGH | Corbetta & Shulman, 2002 |
| Mind wandering and attention withdrawal | 0.22 | LOW | Buckner et al., 2008 |
| Involuntary attention capture | 0.61 | MODERATE | Corbetta & Shulman, 2002 |
| Cognitive effort and working memory | 0.58 | MODERATE | Vincent et al., 2008 |
| Visual processing | 0.81 | VERY HIGH | Wandell et al., 2007 |
| Auditory and speech processing | 0.49 | MODERATE | Formisano et al., 2003 |
| Language comprehension | 0.55 | MODERATE | Fedorenko et al., 2011 |

Interpret the fields as follows:

- Activation is the aggregated model output for the parcels assigned to that network.
- Level is a thresholded label derived from the relative distribution across parcels in the same run.
- Source names the paper used to justify the network label, not a paper that validated this app as a clinical measure.

The brain image shows predicted cortical activation across both hemispheres. It is a visualization of model output, not a medical scan.

---

## Limits

These outputs are in silico predictions from a population trained model.

They do not tell you:
- what a given individual subject actually felt
- whether a diagnosis is present
- whether a piece of content is universally "good"
- whether the model captured causation rather than correlation

They do tell you:
- what cortical response pattern the model predicts for the supplied input
- how that pattern distributes across the selected parcel groups
- which attention related systems appear more active than others within that run

---

## Attention networks explained

**Voluntary attention control (Dorsal Attention Network)**  
Use this label when the predicted response is elevated in parcels associated with deliberate top down attention. A higher value suggests the content sustained directed tracking or selection. Source: Corbetta & Shulman, 2002.

**Mind wandering and attention withdrawal (Default Mode Network)**  
Use this label when parcels associated with internally directed processing are more active. A higher value suggests weaker alignment with externally driven task focus. Source: Buckner et al., 2008.

**Involuntary attention capture (Ventral Attention Network)**  
Use this label when parcels associated with stimulus driven reorienting are more active. A higher value suggests salience, surprise, or interruption of an ongoing focus state. Source: Corbetta & Shulman, 2002.

**Cognitive effort and working memory (Frontoparietal Control Network)**  
Use this label when parcels associated with control, holding information, and task switching are more active. A higher value suggests higher cognitive demand. Source: Vincent et al., 2008.

**Visual processing**  
Use this label when early and intermediate visual parcels are more active. A higher value usually tracks visual complexity, motion, contrast, or scene change. Source: Wandell et al., 2007.

**Auditory and speech processing**  
Use this label when auditory and speech selective parcels are more active. A higher value usually tracks spoken language, music, or dense acoustic structure. Source: Formisano et al., 2003.

**Language comprehension**  
Use this label when left lateralized language related parcels are more active. A higher value suggests stronger semantic and sentence level processing. Source: Fedorenko et al., 2011.

---

## Supported inputs

| Input type | Supported | Reason |
|-----------|-----------|--------|
| YouTube link | Yes | Downloaded and converted into model ready input |
| Video file (.mp4, .avi, .mkv, .mov) | Yes | Passed to TRIBE v2 pipeline |
| Audio file (.wav, .mp3, .flac) | Yes | Passed to TRIBE v2 pipeline |
| Text file (.txt) | Yes | Passed to TRIBE v2 text pathway |
| Instagram or TikTok links | No | Platform access restrictions |
| Images | No | No image only encoder in this setup |

---

## Setup: run locally on your machine

### Requirements

- Python 3.11 or higher
- Mac, Linux, or Windows
- At least 16GB RAM
- GPU optional but recommended

Expected runtime:
- about 30 to 90 seconds on CPU
- about 10 to 20 seconds on GPU

### 1. Clone this repo and the TRIBE v2 repository

```bash
git clone https://github.com/bhj37193/tribe-lens-attention
cd tribe-lens-attention

git clone https://github.com/facebookresearch/tribev2
cd tribev2
pip install -e ".[plotting]"
cd ..
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Accept model terms and download weights

Weights download from HuggingFace on first run. The first download is roughly 8 to 15GB.

Accept the model terms at:
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

Default local URL:
`http://localhost:7860`

---

## Setup: deploy your own HuggingFace Space

### 1. Create a HuggingFace account and upgrade to Pro

Required for ZeroGPU access.

Pricing page:
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

Go to:
https://huggingface.co/facebook/tribev2

Accept the terms so the Space can download the weights.

### 5. Add your HuggingFace token as a Space secret

In Space settings, add a secret named `HF_TOKEN`.

The live URL will be:
`https://huggingface.co/spaces/YOUR_USERNAME/tribe-lens-attention`

---

## Cost to run

### Local

**$0/month** after you use your own machine and internet connection for the initial model download.

### HuggingFace Space

| Item | Cost |
|------|------|
| HuggingFace Pro subscription | $9/month |
| ZeroGPU compute | Included in Pro |
| Spaces hosting | Included in Pro |
| Model weight storage | Free from HuggingFace Hub |
| yt dlp | Free |

For personal use, that puts the recurring platform cost at **$9/month**.

If you expect concurrent heavy use, a dedicated GPU is the more realistic option.

---

## Project structure

```text
tribe-lens-attention/
├── app.py                  # Gradio interface
├── tribe_inference.py      # Loads model and runs predict()
├── attention_mapper.py     # Maps HCP parcels to attention networks
├── brain_plot.py           # Renders brain surface image
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How it works technically

1. The app accepts either a YouTube URL or a local file.
2. The input is passed to `TribeModel.get_events_dataframe()`.
3. TRIBE v2 converts the input into model events through extraction and transcription steps when needed.
4. `TribeModel.predict()` returns a tensor with shape `(n_segments, 20484)`.
5. The app averages across segments to get one `(20484,)` cortical prediction vector.
6. `summarize_by_roi()` in `tribev2/utils.py` averages vertices inside each HCP MMP1.0 parcel.
7. The app groups selected parcels into seven named systems.
8. It converts relative magnitudes into LOW, MODERATE, HIGH, or VERY HIGH labels.
9. `PlotBrainNilearn` renders the surface map.
10. The app returns the image and the summary table.

Every stage above can be inspected in code. The app is not using a hidden rubric such as "viral content" or "good content."

---

## Built on

- **TRIBE v2** by Meta Research. d'Ascoli et al., 2026. *A Foundation Model of Vision, Audition, and Language for In Silico Neuroscience*. https://huggingface.co/facebook/tribev2
- **HCP MMP1.0 Parcellation** by Glasser et al., 2016. *A multi modal parcellation of human cerebral cortex*. Nature, 536, 171 to 178.
- **Nilearn** by Abraham et al., 2014. *Machine learning for neuroimaging with scikit learn*. Frontiers in Neuroinformatics.

---

## License

This project is licensed under **CC BY NC 4.0**.

You may:
- use the code for personal, research, or educational work
- share and adapt it with attribution
- deploy your own non commercial instance

You may not:
- use it for commercial purposes
- sell access to the tool
- remove attribution to TRIBE v2 or this repository

Full license text:
https://creativecommons.org/licenses/by-nc/4.0/

If you use the project in research or in a public project, cite both TRIBE v2 and this repository.

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

Issues and pull requests are welcome.

Contributions should preserve:
- the research framing
- the non commercial license terms
- attribution to upstream work

---

Built by [@bhj37193](https://github.com/bhj37193)
