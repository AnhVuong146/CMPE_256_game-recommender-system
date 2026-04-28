# Board Game Recommender System
**CMPE 256 — Recommender Systems | Team 10**

Dataset: [Board Games Database from BoardGameGeek](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek) · Spring 2026

---

## Overview

This project designs, implements, and evaluates three advanced board game recommender systems built on the BoardGameGeek (BGG) dataset — 411K users, 22K games, and 19M explicit 1-10 ratings. All three variants are built on a shared preprocessing pipeline and evaluated on the same test set using RMSE, MAE, Precision@10, and Recall@10.

---

## Repository Structure

```
├── eda.ipynb                        # Exploratory data analysis — sparsity, cold start, long tail
├── preprocessing.ipynb              # Feature engineering — drop columns, combine metadata
├── baselines.ipynb                  # Shared pipeline + Popularity and KNN baselines
├── LightGCN_AnhVuong.ipynb          # Variant A — Graph-based recommendation (LightGCN)
├── hybrid_neuMF_xgb_ethan.ipynb              # Variant B — Hybrid Neural Matrix Factorization + XGBoost
├── LLM-based_rating_predictor.ipynb            # Variant C — LLM-based rating predictor (GPT-2)
└── README.md
```

---

## Team

| Member | Variant |
|---|---|
| Anh Vuong | Graph-Based Recommendation (LightGCN) |
| Ethan Ho | Hybrid Neural Matrix Factorization (NeuMF + XGBoost) |
| Nicholas Bao | LLM-Based Rating Predictor (GPT-2) |

---

## Setup

**Requirements**
- Python 3.9+
- PyTorch
- scikit-learn
- pandas, numpy, scipy
- xgboost
- transformers (Hugging Face)
- kagglehub

Install all dependencies:
```bash
pip install torch scikit-learn pandas numpy scipy xgboost transformers kagglehub
```

**Dataset**

The dataset is downloaded automatically via `kagglehub` when you run any notebook. No manual download needed. Make sure you have a Kaggle account and your API key configured:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## How to Reproduce Results

Run notebooks in this order:

```
1. eda.ipynb
2. preprocessing.ipynb
3. baselines.ipynb         
4. LightGCN_AnhVuong.ipynb
5. hybrid_neuMF_xgb_ethan.ipynb
6. LLM-based_rating_predictor.ipynb
```

All variants use the same `random_state=42` split from `baselines.ipynb`. The `%store` magic in `preprocessing.ipynb` passes the cleaned metadata to downstream notebooks.

---

## Results

All models evaluated on the same 167,327 test pairs (15K-user KNN/LightGCN subset). Relevance threshold: rating ≥ 7.

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Popularity (per-game mean) | 1.3145 | 0.9930 | 0.0366 | 0.0672 |
| User-Based KNN (k=40) | 1.3072 | 0.9980 | 0.0022 | 0.0042 |
| LightGCN — Anh (L=3, d=64) | 1.2180 | 0.9141 | 0.0222 | 0.0309 |
| XGB + NeuMF — Ethan | 1.0572 | — | 0.1370 | 0.9981 |
| LLM GPT-2 — Nicholas | 3.5262 | — | 0.6000 | 0.6000 |

---

## Variant Summaries

### Variant A — LightGCN (Anh Vuong)
Implements LightGCN (He et al., SIGIR 2020) over a bipartite user-game graph. Builds the normalized adjacency matrix from scratch in PyTorch without any GNN library. Three layers of graph convolution propagate embeddings across multi-hop neighborhoods, enabling the model to find indirect connections between users and games despite 99.79% sparsity. Prediction uses dot product of 64-dim user and game embeddings plus per-user and per-game bias terms. Trained with MSE loss, Adam (lr=0.001), batch size 65,536, early stopping patience=10.

### Variant B — Hybrid NeuMF + XGBoost (Ethan Ho)
Two-stage hybrid: NeuMF (He et al., 2017) learns collaborative latent factors via a GMF tower and a 4-layer MLP tower. The trained user embeddings, item embeddings, and predicted scores are then concatenated with game metadata and fed into an XGBoost regressor. XGBoost was chosen to address NeuMF's overfitting on sparse data. Achieves the best RMSE across all variants.

### Variant C — LLM Rating Predictor (Nicholas Bao)
Uses GPT-2 Large (via Hugging Face) to predict ratings zero-shot from structured game metadata converted into natural language prompts. Prompts include game title, mechanics, complexity, and themes. No fine-tuning on BGG data. Achieves strong ranking metrics but high RMSE due to the model's lack of calibration to the BGG 1-10 rating scale.
