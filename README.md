# Pokémon Pixel-Art Diffusion (from scratch)

A hand-written **DDPM** (denoising diffusion probabilistic model) that generates **96×96 Pokémon pixel art** — built from scratch with `torch` only (no `diffusers`). You can generate a Pokémon **by name**, **by free-text description**, or **fuse two** into a new creature.

The project was built iteratively across **9 experiments**, from a tiny name-conditioned UNet up to a **CLIP-text + cross-attention** model.

## What's in here

- **[`diffusion_sharing.ipynb`](diffusion_sharing.ipynb)** — the main teaching / demo notebook:
  - a live **convolution animation** (a 3×3 window sweeping a Pikachu sprite),
  - **self-attention** and **cross-attention** ablations (real *with vs. without* comparisons),
  - generation **by name** & **by free text**, plus **"create your own Pokémon"**,
  - **word clouds** of the fixed description vocabulary (color / type / shape / generation).
- **`pokemon_train_xattn.ipynb`** — training script for the cross-attention UNet (exp08/09 lineage).
- **`experiments/expNN_…/`** — per-experiment generation notebooks + `classes.json`.
- **`dataset_prep/`** — PokéAPI fetch, augmentation, re-centering, CLIP feature pre-compute, ablation scripts.
- **`lecture_assets/`** — figures used by the notebooks.

## Model weights → Hugging Face

Checkpoints are too large for git, so they live on the Hub:

**🤗 https://huggingface.co/WEIEE/pokemon-diffusion**

```python
from huggingface_hub import hf_hub_download
import torch

ckpt = hf_hub_download(
    "WEIEE/pokemon-diffusion",
    "experiments/exp09_stage1_pokemonALLcentered_xattn48/checkpoints/ckpt_ep300.pt",
)
state = torch.load(ckpt, map_location="cpu")   # model classes are defined in the notebooks
```

## Experiments

| exp | data | conditioning | attention | classes | params |
|----:|------|--------------|-----------|--------:|-------:|
| 01 | all gens, front sprites | name embedding | – | 1070 | 4.2M |
| 02 | 20 classics (augmented) | name embedding | – | 20 | 3.9M |
| 03 | 20 classics (augmented) | name embedding | self-attn | 20 | 5.8M |
| 04 | 20 classics (clean) | name embedding | self-attn + EMA | 20 | 30.3M |
| 05 | 20 classics (augmented) | name embedding | self-attn + EMA | 20 | 30.3M |
| 06 | all 988 Pokémon | CLIP **pooled** text | – | 988 | 30.5M |
| 07 | all 988 Pokémon | CLIP **per-token** text | self + cross-attn @24²/12² | 988 | 33.5M |
| 08 | all 988 Pokémon | CLIP **per-token** text | self + cross-attn @48²/24²/12² | 988 | 34.0M |
| 09 | all 988, **re-centered** | CLIP **per-token** text | self + cross-attn @48²/24²/12² | 988 | 34.0M |

Ideas explored along the way: **classifier-free guidance**, **EMA** (with warmup — critical on small data), **SLERP** fusion (norm-preserving), **two-stage training with rehearsal** (avoid catastrophic forgetting), **silhouette-centered data** (exp09), and **cross-attention** for spatial text control.

## Notes

- The notebooks were developed on a rented GPU server and reference `/root/autodl-tmp/...` paths. To run elsewhere, pull the weights from the Hugging Face link above and point the paths there.
- Datasets (sprites) and the Python venv are intentionally **not** committed — see [`.gitignore`](.gitignore).
- Educational / research project. Pokémon and all sprites are © Nintendo / Game Freak.
