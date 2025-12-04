# Topic Modeling & Clustering Dashboard

An interactive web dashboard for exploring topic distributions and document clusters in the 20 Newsgroups dataset using LDA topic modeling and K-Means clustering.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?logo=typescript)

---

## What It Does

1. **Data Preprocessing**: Loads the 20 Newsgroups dataset and applies text preprocessing (tokenization, lemmatization, stopword removal) using spaCy
2. **Topic Modeling**: Trains LDA (Latent Dirichlet Allocation) models using Gensim for topic counts ranging from 2-20
3. **Dimensionality Reduction**: Applies UMAP to project high-dimensional document-topic distributions to 2D for visualization
4. **Clustering**: Performs K-Means clustering on document-topic distributions
5. **Interactive Dashboard**: Provides a React-based UI to explore topics, clusters, and quality metrics in real-time

---

## Features

- **LDA Topic Modeling** with coherence scores
- **K-Means Clustering** with silhouette scores and elbow detection

- **Interactive Charts**: Coherence plot, silhouette/inertia plot, token distribution box plots
- **UMAP Scatter Plot**: 2D visualization of clustered documents
- **Topic Word Lists**: Top words per topic with probabilities
- **pyLDAvis Integration**: Interactive topic-word distribution explorer
- **Custom Stopwords**: Add domain-specific stopwords via text file
- **Precomputation**: All models precomputed for instant slider response

---

## Tech Stack

**Backend**: FastAPI, Gensim, scikit-learn, spaCy, UMAP-learn, pyLDAvis

**Frontend**: React 18, TypeScript, Vite, Plotly.js, Tailwind CSS v4, SWR

**Package Managers**: uv (Python), bun (JavaScript)

---

## Repository Structure

```
topic-modeling-clustering/
├── backend/
│   ├── api/              # FastAPI route handlers
│   ├── core/             # ML modules (LDA, clustering, UMAP)
│   ├── cache/            # Cache management
│   ├── models/           # Pydantic request/response models
│   └── app.py            # FastAPI application
├── frontend/src/
│   ├── api/              # API client
│   ├── components/       # React components (Dashboard, Charts, etc.)
│   ├── hooks/            # Custom React hooks with SWR
│   └── types/            # TypeScript types
├── scripts/
│   ├── eda.py            # EDA pipeline (token distribution analysis)
│   └── precompute.py     # Model training pipeline
├── tests/                # pytest unit tests
├── cache/                # Cached artifacts (git-ignored)
├── custom_stopwords.txt  # User-defined stopwords
└── main.py               # Backend entry point
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+ (or Bun)
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Install Dependencies

```bash
# Python dependencies
# Clone the repository
git clone <repo-url>
cd topic-modeling-clustering
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync

# Frontend dependencies
cd frontend && bun install && cd ..
```

### 2. Run Precomputation

**Option A: With EDA (Recommended)**

Analyzes token distributions and filters low-quality documents before training.

```bash
# Run EDA and save filtered data
uv run python scripts/eda.py --save-filtered

# Train models on filtered data
uv run python scripts/precompute.py --use-filtered
```

**Option B: Direct**

```bash
# Train on all documents (no filtering)
uv run python scripts/precompute.py
```

Full precomputation takes ~30-90 min. For quick testing:

```bash
uv run python scripts/precompute.py --min-topics 2 --max-topics 4
```

### 3. Start the Application

```bash
# Terminal 1 - Backend
uv run main.py

# Terminal 2 - Frontend
cd frontend && bun run dev
```

Open http://localhost:5173

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=backend
```

---

## License

MIT License
