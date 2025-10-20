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
│── 📂 encoders
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

## 📚 Future Work

* Add more advanced backbone architectures (Transformers, GNNs).
* Extend cross-feature interactions.
* Provide pre-trained encoders for reproducibility.
* **SQL Integration**:

  * Store extracted features and results in relational databases.
  * Use SQL queries for efficient filtering, aggregation, and joining of RNA/disease datasets.
  * Enable hybrid pipelines where SQL preprocessing feeds into encoder/feature modules.

---

