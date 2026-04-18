# ML Coursework — University of Georgia (CSCI 8950)

Graduate-level machine learning labs from **CSCI 8950** at the University of Georgia (Spring 2026).  
Each lab is a standalone Jupyter notebook covering a different ML topic with working implementations.

---

## Labs

### Lab 10 — Embeddings, Recommender Systems & Neural Network Variants
**[`lab10-embeddings-recommenders/lab10.ipynb`](./lab10-embeddings-recommenders/lab10.ipynb)**

| Topic | Details |
|---|---|
| **Embeddings** | One-hot encoding vs dense embeddings, embedding lookup from scratch |
| **Similarity** | Cosine similarity, top-k nearest neighbor retrieval |
| **Recommenders** | Matrix factorization with SGD, dot-product neural model, MLP collaborative filtering |
| **Evaluation** | RMSE, MAE, train/valid split, early stopping, L2 regularization |
| **Visualization** | PCA and t-SNE of learned movie embedding spaces |
| **Stack** | Python, NumPy, Pandas, TensorFlow/Keras, Scikit-learn, Matplotlib |

**Key results:** MLP with dropout achieved lowest validation RMSE; PCA of learned embeddings showed clear semantic genre clustering.

---

## Setup

```bash
git clone https://github.com/Firozkhan009/ml-coursework.git
cd ml-coursework
pip install numpy pandas matplotlib scikit-learn tensorflow
jupyter notebook
```

> **Dataset:** Labs use the [MovieLens dataset](https://grouplens.org/datasets/movielens/). Download separately and place in the `data/` folder — not tracked in git.

---

## About

**Firoz Khan Patan** | MS Computer Science, University of Georgia  
[LinkedIn](https://linkedin.com/in/firoz-khan-patan) · [GitHub](https://github.com/Firozkhan009) · [firozkhanp009@gmail.com](mailto:firozkhanp009@gmail.com)
