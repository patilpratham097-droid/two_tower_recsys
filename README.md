# Hybrid Two-Tower Recommendation System

A PyTorch-based recommendation engine built from scratch to explore production-grade architecture patterns used by modern tech platforms. This project pairs collaborative filtering behavioral logs with explicit content metadata (movie genres) using a decoupled Two-Tower neural network.

---

## Why a Two-Tower Architecture?

In production environments with millions of users and items, scoring every single item against a user via a monolithic neural network introduces massive latency. 

This architecture splits the model into two independent paths:
* **The User Tower:** Processes user IDs and historical interaction patterns into a dense latent vector ($v_u$).
* **The Item Tower:** Fuses movie ID embeddings with 19-dimensional binary genre flags via tensor concatenation, projecting them into a matching latent vector space ($v_i$).

**The Engineering Benefit:** At inference time, item vectors are pre-computed offline and stored in memory. When a user requests recommendations, only the User Tower runs online, and recommendations are generated instantly via a lightweight matrix dot product (`torch.sum`) across all items.

---

## Project Structure

```text
two_tower_recsys/
├── data/                  # MovieLens 100K raw files (git-ignored)
├── models/                # Serialized model checkpoints (.pth) (git-ignored)
├── src/
│   ├── dataset.py         # Custom PyTorch Dataset, binarization, & genre extraction
│   ├── model.py           # Two-Tower neural network (Embedding MLPs + Dot Product)
│   ├── train.py           # Training loop using BCELoss and Adam optimizer
│   └── recommend.py       # Inference script for fast vector similarity retrieval
├── .gitignore             # Excludes large binaries and dataset dumps
└── README.md              # Project documentation

## Getting Started
1. Clone and Setup
Bash
git clone [https://github.com/patilpratham097-droid/two_tower_recsys.git](https://github.com/patilpratham097-droid/two_tower_recsys.git)
cd two_tower_recsys
pip install torch pandas
2. Train the Model
The training script automatically fetches and extracts the MovieLens 100K dataset, builds the genre matrix, trains for 5 epochs using Binary Cross-Entropy loss, and saves the weights:

Bash
python -m src.train
3. Generate Recommendations
Run inference for any user ID to output top hybrid recommendations complete with readable movie titles:

Bash
python -m src.recommend