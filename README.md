# CAFA6 Protein Function Prediction

This repository adapts the [CAFA5 2nd place solution](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place) for the CAFA6 competition. All modifications, experimental findings, and running instructions are documented below.

---

## CONTENTS

* `nn_solution/` — Neural Network base model training & inference
* `protlib/` — Py-Boost GBDT & Logistic Regression training, data preprocessing, GO metric computation
* `protnn/` — GCN stacker model training & inference
* `CAFA6PIpeline.ipynb` — **Main notebook**: full pipeline with all modifications documented. Follow this notebook step by step to reproduce the solution. Some steps can be skipped using pre-computed files (embeddings, temporal data). Pre-trained model weights and related data are available at https://pan.quark.cn/s/4a4b2b6aae4b. For comparison, see the original `CAFA5PIpeline.ipynb` at the [upstream repo](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place).
* `command.md` — All training/inference commands used in our runs
* `config.yaml` — Model and path configuration (updated for CAFA6 data sizes)
* `check/` — Analysis reports, propagation comparison, environment matrix, data analysis
* `test_addiif/` — IF model experiments (7-model GCN with ESM-IF embeddings)
* `create-rapids-pb-env.sh` — Install RAPIDS 23.02 env for preprocessing & ML
* `create-pytorch-env.sh` — Install PyTorch env for DL models
* `create-rapids-gcn-env.sh` — Install RAPIDS 26.06 env for postprocessing
* `CAFA5docs.pdf` — Original solution description (CAFA5)

---

## KEY MODIFICATIONS FOR CAFA6

### Data Preprocessing
| Change | Details |
|------|------|
| **GO DAG propagation** | Enabled `--propagate True` in `create_helpers.py`. GCN SWA best improved: BP +0.073, MF +0.037, CC +0.009 |
| **Deduplication + binarization** | Added `drop_duplicates` + `trg[trg>0]=1.0` to prevent label values >1 |
| **Ontology-parallel** | Added `--ontology` flag for 3x faster parallel execution |
| **CAFA6 data merge** | CAFA6 (82,404 proteins) as primary + CAFA5 unique (62,978) as supplement = 145,382 total |

### Model Training
| Change | Details |
|------|------|
| **LogReg alpha** | Changed from 1e-5 → 1e-7 for fewer NaN convergence failures on sparse conditional terms |
| **LogReg parallel folds** | Added `--folds` option + `merge_folds.py` for parallel 5-fold CV |
| **NN data sizes** | Updated `train_models.py`/`inference_models.py`: train 142246→145382, test 141865→224309 |
| **prepare.py** | Added CAFA6 aspect code compatibility (P/F/C + BPO/MFO/CCO) and NaN row filtering |

### GCN & Inference
| Change | Details |
|------|------|
| **Memory fix** | Added `del` + `gc.collect()` between TTA configs AND ontologies in `predict_gcn.py`, which may reduce peak RAM usage vs. the original code |
| **Inference memory** | Requires **124GB RAM** for full 4-TTA prediction (original 62GB caused cgroup OOM) |
| **num_workers** | Adjusted to 8 (stable at 124GB). WARNING header added to `predict_gcn.py` |
| **CC-only prediction** | Script at `aaa/ccadd/predict_cc_only.py` for separate CC ontology inference |
| **BP checkpoint resume** | `test_addiif/train_gcn_resume.py` for resuming from SWA checkpoint |
| **IF experiments** | 7-model GCN with ESM-IF embeddings (`test_addiif/train_gcn_if.py`): CC +0.005, MF -0.003 |
| **Hidden size experiments** | Tested hidden=24/32: slower convergence, worse than hidden=16 |

### Environment
| Environment | Python | CUDA | RAPIDS | Purpose |
|------|------|------|------|------|
| `pytorch-env` | — | 12 | — | NN training, GCN training/inference |
| `rapids-pb-env` | 3.8 | 11.2 | 23.02 | Data prep, py-boost, LogReg training |
| `rapids-gcn-env` | 3.12 | 12 | 26.06 | GCN epoch eval, postprocessing (cudf) |

---

## HARDWARE (Our Setup)

| Component | Spec |
|------|------|
| GPU | 4× NVIDIA RTX 4080 Super (32GB VRAM each) |
| CPU | 16 vCPU Intel Xeon Platinum 8352V @ 2.10GHz |
| RAM | 62GB (training), 124GB (inference — **critical** for 4-TTA prediction) |
| Disk | 30GB system + 1TB data |

### Original CAFA5 Hardware (for reference)
* 2× Tesla V100 32GB, 512GB RAM
* Training times  faster on  RTX 4080s 

---

## SOFTWARE

* Ubuntu 22.04
* Python ≥3.8 for notebook execution (only `pyyaml` required)
* Conda ≥23.5.2 with libmamba solver
* NVIDIA driver supporting CUDA 12

---

## QUICK START

### 1. Setup environments
```bash
./create-rapids-pb-env.sh .
./create-pytorch-env.sh .
./create-rapids-gcn-env.sh .
```

### 2. Run the pipeline
Open `CAFA6PIpeline.ipynb` and execute cells step by step. Pre-computed files are provided for:
- `./embeds/` — T5 (1024d), ESM2-650M (1280d), ESM-IF (512d)
- `./temporal/` — GOA electronic annotations (already propagated)

### 3. Key requirements
- **Training:** 62GB+ RAM, RTX 4080 (or equivalent 32GB GPU)
- **Inference:** **124GB RAM required** for full 4-TTA GCN prediction
- **Disk:** ~50GB for model weights + embeddings

---

## TRAINING TIMES (RTX 4080 Super)

| Stage | Time (4 GPUs) |
|------|------|
| Data preparation + embeddings | ~3 hours |
| Base models (6 GBDT + 2 LogReg + 1 NN) | ~18 hours |
| GCN training (BP ~10.7h, MF ~3h, CC ~1.3h) | ~13 hours |
| GCN inference (4 TTA × 3 ontologies) | ~2.5 hours |
| Postprocessing | ~10 minutes |
| **Total** | **~37 hours** |

---

## GCN SCORES

| Ontology | Without propagate | With propagate | Gain |
|------|------|------|------|
| BP | 0.3275 | 0.4004 | **+0.073** |
| MF | 0.6636 | 0.7048 | **+0.037** |
| CC | 0.5983 | 0.6074 | **+0.009** |

With IF features (7-model GCN, hidden=16):
| Ontology | 5-model (original) | 7-model (+IF) | Δ |
|------|------|------|------|
| CC | 0.6015 | 0.6069 | +0.005 |
| MF | 0.7048 | 0.7020 | -0.003 |

---

## REFERENCES

* Original solution: https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place
* CAFA5docs.pdf for detailed methodology
