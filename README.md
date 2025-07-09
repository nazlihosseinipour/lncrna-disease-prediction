# GIP Kernel Similarity – Toy Example

This project demonstrates how to compute **Gaussian Interaction Profile (GIP) kernel similarity** for both lncRNAs and diseases using a simple toy matrix.

## What it does

- Generates a random binary interaction matrix (lncRNA–disease)
- Computes pairwise distances between lncRNAs and between diseases
- Applies the **GIP kernel** to convert distances into similarity scores
- Displays the similarity matrices using labeled tables

## Why this matters

This is a first step in understanding how feature engineering works in lncRNA–disease association prediction models like IPCARF.

## How to run it

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```
3. Run the script:

```bash
python gip_similarity.py
```
