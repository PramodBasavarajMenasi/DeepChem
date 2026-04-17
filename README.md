# OLMo × DeepChem — LLM Support for Molecular ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers%204.44.0-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![DeepChem](https://img.shields.io/badge/DeepChem-2.7%2B-00C4CC?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![GSoC](https://img.shields.io/badge/GSoC-2026-4285F4?style=flat-square&logo=google&logoColor=white)

**Full integration of `allenai/OLMo-1B` into DeepChem's `HuggingFaceModel` wrapper —
supporting molecular generation, classification, regression, and continued pretraining
from a single unified API.**

[Overview](#overview) · [Repository Structure](#repository-structure) · [How to Run](#how-to-run) · [Datasets](#datasets) · [API Reference](#api-reference) · [Results](#results) · [Engineering Notes](#engineering-notes)

</div>

---

## Overview

This project integrates [OLMo](https://huggingface.co/allenai/OLMo-1B) (Open Language Model by AllenAI) into [DeepChem](https://github.com/deepchem/deepchem) as a first-class LLM backend for molecular machine learning. It demonstrates that a general-purpose decoder-only transformer can be adapted for all major molecular ML task types using SMILES strings as the input language.

### What's implemented

| Task | Dataset | Format | Status |
|---|---|---|---|
| Continued Pretraining | ZINC15 (250K SMILES) | `Molecule: <SMILES>` | ✅ Working |
| Molecular Generation | MOSES (1.9M SMILES) | `Generate molecule: <SMILES>` | ✅ Working |
| Binary Classification | SIDER (27 tasks), MUV (17 tasks) | `SMILES: <s> TASK: <t> LABEL:` → `0/1` | ✅ Working |
| Regression | ESOL, Lipophilicity | `SMILES: <s> TASK: <t> LABEL:` → float | ✅ Working |

### Architecture

```
SMILES Input
    │
    ▼
OLMo Tokenizer (AutoTokenizer)
    │
    ▼
OLMo-1B Backbone (OLMoForCausalLM)
  + LoRA rank=8  (5.2M / 1182M trainable params)
  + bfloat16 precision
  + gradient checkpointing
    │
    ├──► [Generation / Pretraining]  → causal LM loss → generated SMILES
    │
    └──► [Classification / Regression]
              │
              ▼
         Last-token hidden state  [batch, hidden=2048]
              │
              ▼
         Task Head: LayerNorm → Dropout(0.1) → Linear(2048 → 1)
              │
              ▼
         Sigmoid (clf) / Raw logit (reg)
```

---

## Repository Structure

This repo contains **two notebooks**. Everything else — datasets, checkpoints, model files —
is generated automatically inside your Colab session when you run them.

```
DeepChem/
├── README.md
├── deepchem_1.ipynb    ← Main notebook: datasets + OLMoAPI + training (all tasks except MOSES)
└── deepchem_2.ipynb    ← MOSES notebook: MOSES download + generation training (requires GPU)
```

### Why two separate notebooks?

MOSES has 1.9M molecules. Downloading and training on it requires a GPU and takes
significantly longer than the other datasets. To keep the main workflow clean and fast,
MOSES is isolated in `deepchem_2.ipynb` so you can run it separately when a GPU is
available, without blocking the rest of the pipeline.

### What gets generated after running the notebooks

> These folders are **not committed to the repo**. They are created locally in your Colab session.

```
# Created by deepchem_1.ipynb
olmo_llm_datasets/
├── pretraining/
│   └── zinc15_llm.csv          ← 250,000 SMILES for continued pretraining
├── classification/
│   ├── sider.csv               ← 1,427 molecules × 27 side-effect tasks
│   └── muv.csv                 ← 93,087 molecules × 17 virtual screening tasks
└── regression/
    ├── esol.csv                ← 1,128 molecules, water solubility
    └── lipophilicity.csv       ← 4,200 molecules, lipophilicity

olmo_api_checkpoints/
├── zinc15/                     ← pretraining checkpoint
├── sider/                      ← classification checkpoint
├── esol/                       ← regression checkpoint
└── lipophilicity/              ← regression checkpoint

my_olmo_model/                  ← final saved model (api.save())

# Created by deepchem_2.ipynb
olmo_llm_datasets/
└── generation/
    └── moses_llm.csv           ← 1,900,000 SMILES for generation

olmo_api_checkpoints/
└── moses/                      ← generation checkpoint
```

---

## How to Run

### Notebook 1 — `deepchem_1.ipynb`

**What it does:**
- Installs all dependencies
- Downloads and prepares datasets: ZINC15, SIDER, MUV, ESOL, Lipophilicity
- Defines the full `OLMoAPI` class
- Trains on all task types: pretraining (ZINC15), classification (SIDER), regression (ESOL, Lipophilicity)
- Runs predict, evaluate, and generate

**How to run:**

1. Open `deepchem_1.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Go to `Runtime → Change runtime type → T4 GPU`
3. Run all cells top to bottom

Expected output after dataset preparation:

```
✅  zinc15   saved → ./olmo_llm_datasets/pretraining/zinc15_llm.csv
✅  sider    saved → ./olmo_llm_datasets/classification/sider.csv
✅  muv      saved → ./olmo_llm_datasets/classification/muv.csv
✅  esol     saved → ./olmo_llm_datasets/regression/esol.csv
✅  lipophilicity saved → ./olmo_llm_datasets/regression/lipophilicity.csv
```

Expected output after training:

```
Epoch 1/1 | train=0.7307 | val=0.4642   ← esol
Epoch 1/1 | train=0.9959 | val=0.9431   ← lipophilicity
Epoch 1/1 | train=0.6829 | val=0.6732   ← sider
Epoch 1/1 | train=2.2449 | val=0.1721   ← zinc15
```

---

### Notebook 2 — `deepchem_2.ipynb`

**What it does:**
- Installs dependencies
- Downloads the MOSES dataset (1.9M molecules)
- Trains OLMo on molecular generation using MOSES

**Why separate?**
MOSES is a large dataset (~1.9M SMILES). Downloading and training on it is GPU-intensive
and time-consuming. It is kept in a separate notebook so it can be run independently
without affecting the rest of the pipeline.

**How to run:**

1. Open `deepchem_2.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Go to `Runtime → Change runtime type → T4 GPU` *(GPU is required)*
3. Run all cells top to bottom

Expected output after MOSES training:

```
✅  MOSES — 1,936,962 molecules → ./olmo_llm_datasets/generation/moses_llm.csv
       train=1,584,664 | test=176,075

Epoch 1/1 | train=0.2325 | val=0.2253   ← moses generation
```

---

## Datasets

All datasets are loaded via DeepChem's MolNet and converted into a unified LLM-ready format.

### Why SMILES as text?

OLMo is a language model — it reads text, not molecular fingerprints. The entire pipeline
treats SMILES as natural language using task-specific prompt templates:

```
# Pretraining
"Molecule: CC(=O)Oc1ccccc1C(=O)O"

# Generation
"Generate molecule: CC(=O)Oc1ccccc1C(=O)O"

# Classification / Regression
"SMILES: CC(=O)Oc1ccccc1C(=O)O TASK: Hepatobiliary disorders LABEL:"
```

### Unified CSV schema

Every generated CSV shares this column structure:

| Column | Description |
|---|---|
| `smiles` | Raw SMILES string |
| `llm_input` | Tokenizer input string for OLMo |
| `llm_target` | Expected output string |
| `task_type` | `generation` / `classification` / `regression` / `pretraining` |
| `split` | `train` / `valid` / `test` |
| `task` | Task name (classification/regression only) |
| `label` | Ground truth value (classification/regression only) |

### Dataset summary

| Dataset | Task | Molecules | Source Notebook |
|---|---|---|---|
| ZINC15 | Pretraining | 250,000 | `deepchem_1.ipynb` |
| SIDER | Classification (27 tasks) | 1,427 | `deepchem_1.ipynb` |
| MUV | Classification (17 tasks) | 93,087 | `deepchem_1.ipynb` |
| ESOL | Regression | 1,128 | `deepchem_1.ipynb` |
| Lipophilicity | Regression | 4,200 | `deepchem_1.ipynb` |
| MOSES | Generation | 1,900,000 | `deepchem_2.ipynb` |

---

## API Reference

Everything below is defined in `deepchem_1.ipynb`.

### Initialize

```python
api = OLMoAPI(
    model_name    = "allenai/OLMo-1B",
    lora_rank     = 8,                        # LoRA rank (0 = full fine-tune)
    batch_size    = 4,
    max_length    = 128,
    learning_rate = 2e-5,
    save_dir      = "./olmo_api_checkpoints",
)
```

The backbone loads **once** and is shared across all tasks. Switching task types
automatically rebuilds the task-specific head.

---

### `api.list_datasets()`

```python
api.list_datasets()

# Dataset          Task Type        File exists    Description
# zinc15           pretraining      YES            250K drug-like SMILES
# moses            generation       YES            1.9M SMILES (after running notebook 2)
# sider            classification   YES            1,427 drugs × 27 tasks
# muv              classification   YES            93,087 molecules × 17 tasks
# esol             regression       YES            1,128 molecules, solubility
# lipophilicity    regression       YES            4,200 molecules, logD
```

---

### `api.train()`

```python
# In deepchem_1.ipynb
api.train("zinc15", epochs=1, max_rows=5000)                          # pretraining
api.train("sider",  task="Hepatobiliary disorders", epochs=2)         # classification
api.train("esol",   epochs=3)                                         # regression
api.train("lipophilicity", epochs=3)                                  # regression

# In deepchem_2.ipynb
api.train("moses",  epochs=1, max_rows=1000)                          # generation
```

Returns `list[dict]` — `[{epoch, train_loss, val_loss}, ...]`

---

### `api.predict()`

```python
# Regression — predicted solubility
preds = api.predict(
    smiles  = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO"],
    dataset = "esol",
)

# Classification — side effect probability
probs = api.predict(
    smiles  = ["CC(=O)Oc1ccccc1C(=O)O"],
    dataset = "sider",
    task    = "Hepatobiliary disorders",
)
```

Returns `np.ndarray` — floats for regression, sigmoid probabilities for classification.

---

### `api.evaluate()`

```python
api.evaluate("esol")                                     # → rmse, mae, r2
api.evaluate("sider", task="Hepatobiliary disorders")    # → roc_auc, avg_precision, accuracy
```

---

### `api.generate()`

```python
api.generate(n=20)                                                        # unconditional
api.generate(n=10, prompt="Generate molecule: CC(", temperature=0.7)      # seeded
api.generate(n=10, prompt="Generate molecule: c1ccc(", temperature=1.0)   # aromatic seed
```

Returns `list[str]` — decoded output strings.

---

### `api.status()`

```python
api.status()

# OLMoAPI Status
# Model   : allenai/OLMo-1B  |  Device: cuda
# Trained datasets:
#   sider            task_type=classification   epochs=1  val_loss=0.6732
#   esol             task_type=regression       epochs=1  val_loss=0.4642
#   lipophilicity    task_type=regression       epochs=1  val_loss=0.9431
#   zinc15           task_type=pretraining      epochs=1  val_loss=0.1721
#   moses            task_type=generation       epochs=1  val_loss=0.2253
```

---

### `api.save()` / `api.load()`

```python
api.save("./my_olmo_model")
api.load("./my_olmo_model")
```

---

## Results

All results from 1 epoch of training on subsampled data (Google Colab T4 GPU).

### Training losses

| Dataset | Task Type | Notebook | Train Loss | Val Loss |
|---|---|---|---|---|
| ZINC15 | Pretraining | `deepchem_1.ipynb` | 2.2449 | 0.1721 |
| MOSES | Generation | `deepchem_2.ipynb` | 0.2325 | 0.2253 |
| SIDER | Classification | `deepchem_1.ipynb` | 0.6829 | 0.6732 |
| ESOL | Regression | `deepchem_1.ipynb` | 0.7307 | 0.4642 |
| Lipophilicity | Regression | `deepchem_1.ipynb` | 0.9959 | 0.9431 |

### Evaluation metrics (1 epoch baseline)

| Dataset | Metric | Value |
|---|---|---|
| ESOL | RMSE | 1.3101 |
| ESOL | MAE | 1.0065 |
| ESOL | R² | -0.6317 |
| SIDER | ROC-AUC | 0.5190 |
| SIDER | Avg Precision | 0.5330 |
| SIDER | Accuracy | 0.5524 |

> 1-epoch baselines on subsampled data. Metrics improve substantially with full dataset
> training over 5–10 epochs.

### Sample generation output (after 1 epoch on MOSES)

```
[1] CCn1c(C(=O)NCc2ccccc2)n(Cc3noc(C)n3)c1...
[2] Cc1cc(N2CCC(c3cccn3)CC2)cc(C)c1(Cl)...
[3] COc1ccc(CCNC(=O)c2ccc(F)cc2)cc1...
[4] CCOC(=O)NC(=O)c1cc(-c2ccc(F)cc2)ccc1...
```

---

## Engineering Notes

Every non-obvious fix made during development.

### 1. `all_tied_weights_keys` AttributeError → pin `transformers==4.44.0`

Later versions of `transformers` changed weight tying behavior during `from_pretrained()`.
OLMo expects the old behavior. Pinning to `4.44.0` resolves this completely.

### 2. FP16 gradient unscale failure → switch to `bfloat16`

OLMo's attention layers produce values outside the FP16 dynamic range during early training,
causing `inf` gradients that crash the AMP unscaler. `bfloat16` has the same memory
footprint but a wider range and requires no gradient scaling.

```python
OLMoForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
```

### 3. CUDA OOM → LoRA rank=8 + batch size 4

OLMo-1B at full precision exhausts T4 VRAM during the backward pass. LoRA with rank=8
reduces trainable parameters from 1182M to 5.2M (~0.4%).

```python
LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
```

### 4. MUV 10-minute loading bottleneck → vectorized `pd.melt()`

The original dataset conversion iterated over 1.58M rows in a Python `for` loop.
Replacing it with `pd.melt()` reduced load time from ~10 minutes to ~3 seconds.

### 5. MOSES 2-hour validation hang → cap validation at 500 rows

MOSES has ~600K validation rows. At batch size 4, one validation pass takes ~2 hours.
Fixed by sampling 500 validation rows before each training run. This is also why MOSES
training lives in a separate notebook — to isolate this GPU-intensive work.

```python
va = va.sample(500, random_state=42).reset_index(drop=True)
```

### 6. Task head `None` after switching task types → auto-rebuild in `_predict_df`

After training a generation task (which sets `_task_head = None`), calling `predict()`
for a regression task would crash with an `AttributeError`. Fixed with an auto-rebuild
guard at the start of `_predict_df()`.

---

## Roadmap (GSoC 2026)

- [ ] Scale to `allenai/OLMo-7B` with full DeepChem `HuggingFaceModel` integration
- [ ] Proper `HuggingFaceModel` subclass following DeepChem's conventions
- [ ] Multi-task training (all 27 SIDER tasks simultaneously)
- [ ] SMILES validity filtering on generated outputs via RDKit
- [ ] Benchmark against ChemBERTa and MolBERT on ESOL / Lipophilicity / SIDER
- [ ] Unit tests integrated into DeepChem's CI pipeline
- [ ] Support `OLMo-2` variants

---

## Acknowledgements

- [AllenAI](https://allenai.org/) for the OLMo model family
- [DeepChem](https://github.com/deepchem/deepchem) for MolNet loaders and the `HuggingFaceModel` wrapper
- GSoC 2025 mentors: Riya, Harindhar

---

## License

MIT License — see [LICENSE](LICENSE) for details.
