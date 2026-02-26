# 🧬 RNA Feature Engineering & Encoding Framework

## 📌 Overview

This repository provides a modular **Object-Oriented Programming (OOP) framework** for **RNA representation learning and feature engineering**.
It is designed with flexibility in mind, allowing you to plug in new encoders, feature extractors, and validation tools without rewriting the entire pipeline.

The project is split into two main components:

1. **RNA Encoder System** – Defines abstract and concrete classes for different RNA encoding strategies.
2. **Feature Engineering System** – Provides modular feature extractors (RNA, disease, cross, neural network features) and validation utilities.

---

## 🏗 Project Structure

```
📂 project-root
│── 📂 NeuralNetwork(encoders)
│   ├── abstract_rna_encoder.py    # Abstract base class for RNA encoders
│   ├── aido_rna_encoder.py        # Example implementation of an encoder
│   ├── mp_rna_encoder.py          # Another encoder implementation
│   ├── backbone.py                # Backbone network definitions
│   ├── backbone_registry.py       # Registry for backbone architectures
│
│── 📂 features
│   ├── feature_module.py          # Core feature module manager
│   ├── nn_features.py             # Neural-network-based feature extractors
│   ├── rna_features.py            # RNA sequence-based features
│   ├── disease_features.py        # Disease-specific feature representations
│   ├── cross_features.py          # Cross-domain features (RNA-disease interactions, etc.)
│   ├── validators.py              # Validation logic for inputs/features
│   ├── utils.py                   # Utility functions
│
│── main.py                        # Entry point for running experiments
│── README.md                      # Project documentation

```

---

## 📂 Data Layout

```
Data/
├── raw/             # Source datasets pulled from upstream (not committed cuz it was large)
├── processed/       # Scripts/notebooks used to clean & prepare the data
└── output_data/     # Cleaned CSVs used by the codebase (output of processed, also didin't push  )
final-output/        # Local feature extraction outputs written by scripts (didin't push , large size)
```

Notes:
- Large raw inputs and generated outputs are not pushed to git. Drop your copies into `Data/raw` and rerun the `Data/processed/*.py` helpers to rebuild `Data/output_data`.
- `run_all_rna_features.py` writes per-method outputs into `final-output/` (created automatically).

---

## 🔑 Design Philosophy

### 1. RNA Encoder System

* Uses **abstract base classes** (`abstract_rna_encoder.py`) to define a standard interface.
* Supports multiple encoder implementations (`aido_rna_encoder.py`, `mp_rna_encoder.py`).
* Integrates with pluggable **backbone networks** via a **registry** (`backbone_registry.py`).
* Easy to extend with new encoders.

### 2. Feature Engineering System

* Each feature type is modularized:

  * **RNA features** → `rna_features.py`
  * **Disease features** → `disease_features.py`
  * **Cross features** → `cross_features.py`
  * **NN-based features** → `nn_features.py`
* Central **Feature Module Manager** (`feature_module.py`) handles feature extraction pipelines.
* Validation logic (`validators.py`) ensures data consistency.
* Utility functions (`utils.py`) support reusable tasks.

---

## 🚀 Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run main pipeline

```bash
python main.py
```

### Neural network features (MP-RNA / AIDO) — GPU setup

Prerequisites:
- Install `torch` with GPU support and `transformers` in your environment.
# i coudn't run it so far bcs my laptop doesn't have gpu 
- Allow Hugging Face model access for `yangheng/MP-RNA` and `genbio-ai/AIDO.RNA-1.6B` (or cache them locally).

Env setup example (from repo root):
```bash
export PYTHONPATH=.:mainfolder:NeuralNetwork
```

Smoke tests (small batches; use on a GPU box):
```bash
# MP-RNA sequence embeddings, small batch to reduce VRAM
python mainfolder/main.py nn --method 100 \
  --seqs_csv Data/output_data/sequences_for_oop.csv \
  --batch_size 2 --return_format dataframe \
  -o final-output/mp_seq.csv

# AIDO token embeddings (optionally pick a layer), even smaller batch if needed
python mainfolder/main.py nn --method 104 \
  --seqs_csv Data/output_data/sequences_for_oop.csv \
  --layer 12 --batch_size 1 --return_format dataframe \
  -o final-output/aido_tokens.csv
```

Notes:
- These commands will download the pretrained models on first run unless you point Hugging Face to a local cache.
- Increase `batch_size` only if you have sufficient GPU VRAM.

### Example: Adding a new RNA encoder

1. Create a new file `my_encoder.py` inside `encoders/`.
2. Inherit from `AbstractRNAEncoder`.
3. Implement the required methods (`encode`, `forward`, etc.).
4. Register it in `backbone_registry.py`.

---

## 🧩 Extensibility

This framework is built to be **plug-and-play**:

* Add new encoders without touching existing code.
* Combine RNA, disease, and cross features for richer representation.
* Easily swap backbone architectures.
* Run experiments with different configurations via `main.py`.

---

## Run the tests 
to run the tests , use the comman : 
```bash 
  python -m pytest test_file_name.py
```

---
## 📚 Future Work

* Add more advanced backbone architectures (Transformers, GNNs).
* Extend cross-feature interactions.
* Provide pre-trained encoders for reproducibility.
* **SQL Integration**:

  * Store extracted features and results in relational databases.
  * Use SQL queries for efficient filtering, aggregation, and joining of RNA/disease datasets.
  * Enable hybrid pipelines where SQL preprocessing feeds into encoder/feature modules.

---
