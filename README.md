# Topic Modeling & Clustering Dashboard

An interactive web dashboard for exploring **topic distributions** and **document clusters** in the 20 Newsgroups dataset (~18,000 documents) using LDA topic modeling and K-Means clustering with real-time visualizations.

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.122+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?logo=typescript)

---

## What This Project Does

This project provides an **end-to-end NLP pipeline** for topic modeling and document clustering:

1. **Data Preprocessing**: Loads the 20 Newsgroups dataset and applies text preprocessing (tokenization, lemmatization, stopword removal) using spaCy
2. **Topic Modeling**: Trains LDA (Latent Dirichlet Allocation) models using Gensim for topic counts ranging from 2-20
3. **Dimensionality Reduction**: Applies UMAP to project high-dimensional document-topic distributions to 2D for visualization
4. **Clustering**: Performs K-Means clustering on document-topic distributions
5. **Interactive Dashboard**: Provides a React-based UI to explore topics, clusters, and quality metrics in real-time

---

## Topic Modeling vs Clustering: What's the Difference?

This project uses **two complementary techniques** that are often confused. Here's a clear distinction:

### Topic Modeling (LDA)

**What it does**: Discovers the *themes/subjects* present in documents by analyzing word patterns.

**Output**: Each document gets a **probability distribution over topics**.

| Document | Topic 1 (Sports) | Topic 2 (Politics) | Topic 3 (Tech) |
|----------|------------------|--------------------| ---------------|
| Doc A    | 0.80             | 0.15               | 0.05           |
| Doc B    | 0.10             | 0.70               | 0.20           |
| Doc C    | 0.05             | 0.10               | 0.85           |

**Key insight**: A document can belong to *multiple topics* with different weights. Doc A is 80% sports, 15% politics.

---

### Clustering (K-Means)

**What it does**: Groups *similar documents together* into discrete clusters based on their topic distributions.

**Output**: Each document gets a **single cluster label**.

| Document | Topic Distribution | Cluster |
|----------|-------------------|---------|
| Doc A    | [0.80, 0.15, 0.05] | Cluster 1 |
| Doc B    | [0.10, 0.70, 0.20] | Cluster 2 |
| Doc C    | [0.05, 0.10, 0.85] | Cluster 3 |
| Doc D    | [0.75, 0.20, 0.05] | Cluster 1 |

**Key insight**: Documents are assigned to *exactly one cluster*. Doc A and Doc D are grouped together because they have similar topic distributions.

---

### How They Work Together in This Project

```
                    ┌─────────────────────────────────────────────────┐
                    │              Raw Documents                       │
                    │  "The team won the championship game..."        │
                    │  "Congress passed the new bill today..."        │
                    └───────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────┐
                    │         Step 1: Topic Modeling (LDA)            │
                    │                                                  │
                    │  Discovers latent topics from word patterns:    │
                    │  • Topic 1: game, team, player, score, win      │
                    │  • Topic 2: congress, bill, vote, senator       │
                    │  • Topic 3: computer, software, data, code      │
                    │                                                  │
                    │  Output: Document-Topic Matrix (18K × k)        │
                    │  Each doc → probability over k topics           │
                    └───────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────┐
                    │        Step 2: Clustering (K-Means)             │
                    │                                                  │
                    │  Groups documents with similar topic profiles:  │
                    │  • Cluster A: Mostly sports docs                │
                    │  • Cluster B: Mostly politics docs              │
                    │  • Cluster C: Mixed sports + politics docs      │
                    │                                                  │
                    │  Output: Cluster labels (18K × 1)               │
                    │  Each doc → exactly one cluster ID              │
                    └───────────────────────┬─────────────────────────┘
                                            │
                                            ▼
                    ┌─────────────────────────────────────────────────┐
                    │           Step 3: Visualization (UMAP)          │
                    │                                                  │
                    │  Projects to 2D for scatter plot visualization  │
                    │  Colors = cluster labels                        │
                    └─────────────────────────────────────────────────┘
```

### Simple Analogy

| Concept | Analogy |
|---------|---------|
| **Topic Modeling** | Like analyzing a recipe to find it's 60% Italian, 30% French, 10% Asian based on ingredients |
| **Clustering** | Like grouping recipes into cookbook sections: "Italian Dishes", "Fusion", "Quick Meals" |

**Topic modeling** tells you *what a document is about*.
**Clustering** tells you *which documents are similar to each other*.

---

## Features

### Core ML Features
- **LDA Topic Modeling** with configurable number of topics (2-20)
- **Coherence Score (c_v)** calculation for topic quality evaluation
- **Perplexity Score** calculation for model generalization assessment
- **K-Means Clustering** on document-topic distributions
- **Silhouette Score & Inertia** for cluster quality evaluation
- **Elbow Method** detection for optimal cluster count
- **UMAP Projections** for 2D visualization of document embeddings
- **Custom Stopwords** support via external text file

### Interactive Visualizations
- **Coherence & Perplexity Chart**: Dual y-axis plot for optimal topic selection
- **Silhouette & Inertia Chart**: Dual y-axis plot with elbow point detection
- **UMAP Scatter Plot**: Interactive 2D visualization of clustered documents
- **Topic Word Lists**: Top-10 words per topic with probabilities
- **pyLDAvis Integration**: Interactive topic-word distribution explorer

### Performance Optimizations
- **Precomputation Strategy**: All LDA models precomputed for instant slider response
- **Caching Layer**: Models, projections, and metrics cached to disk
- **Real-time K-Means**: Clustering computed on-demand (<100ms response)
- **Debounced Sliders**: Optimized UI updates with 150ms debounce

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework with async support |
| **Gensim** | LDA topic modeling and coherence metrics |
| **scikit-learn** | K-Means clustering, silhouette scores |
| **spaCy** | Text preprocessing (tokenization, lemmatization) |
| **UMAP-learn** | Dimensionality reduction for visualization |
| **pyLDAvis** | Interactive topic visualization |
| **NumPy** | Numerical computations |
| **Pydantic** | Request/response validation |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 18** | UI component library |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Fast build tool with HMR |
| **Plotly.js** | Interactive charts and scatter plots |
| **Tailwind CSS v4** | Utility-first styling |

### Package Managers
- **uv** - Fast Python package manager
- **bun** - Fast JavaScript runtime and package manager

---

## Repository Structure

```
topic-modeling-clustering/
├── backend/
│   ├── api/                    # FastAPI route handlers
│   │   ├── topics.py           # Topic endpoints (coherence, words, pyLDAvis)
│   │   ├── clustering.py       # Clustering endpoints
│   │   ├── visualization.py    # UMAP projection endpoints
│   │   └── routes.py           # Router aggregation
│   ├── core/                   # Core ML modules
│   │   ├── data_loader.py      # 20 Newsgroups dataset loader
│   │   ├── text_preprocessor.py # spaCy tokenization & lemmatization
│   │   ├── lda_trainer.py      # Gensim LDA training & metrics
│   │   ├── clustering.py       # K-Means clustering wrapper
│   │   ├── projections.py      # UMAP dimensionality reduction
│   │   └── metrics.py          # Silhouette, inertia, elbow detection
│   ├── cache/                  # Cache management utilities
│   │   └── manager.py          # Save/load cached artifacts
│   ├── models/                 # Pydantic request/response models
│   │   ├── requests.py
│   │   └── responses.py
│   ├── app.py                  # FastAPI application factory
│   └── config.py               # Configuration settings
│
├── frontend/
│   └── src/
│       ├── api/
│       │   └── client.ts       # API client with fetch wrapper
│       ├── components/
│       │   ├── Dashboard.tsx   # Main dashboard layout
│       │   ├── ControlPanel.tsx # Topic/cluster sliders
│       │   ├── Charts/
│       │   │   ├── CoherenceChart.tsx    # Coherence + Perplexity plot
│       │   │   └── ClusterMetricsChart.tsx # Silhouette + Inertia plot
│       │   ├── Visualization/
│       │   │   └── ScatterPlot.tsx       # UMAP scatter visualization
│       │   └── Topics/
│       │       ├── TopicWordList.tsx     # Topic words display
│       │       └── PyLDAvisViewer.tsx    # pyLDAvis iframe
│       ├── hooks/              # Custom React hooks
│       │   ├── useTopicData.ts
│       │   ├── useClusterData.ts
│       │   └── useDebounce.ts
│       └── types/
│           └── api.ts          # TypeScript API types
│
├── scripts/
│   └── precompute.py           # Precomputation pipeline script
│
├── tests/                      # pytest unit tests
│   ├── test_api_*.py           # API endpoint tests
│   ├── test_*.py               # Core module tests
│   └── conftest.py             # Shared fixtures
│
├── cache/                      # Cached artifacts (git-ignored)
│   ├── models/                 # LDA models, dictionary, corpus
│   ├── distributions/          # Document-topic matrices
│   ├── projections/            # UMAP coordinates
│   ├── metrics/                # Coherence & perplexity scores
│   └── pyldavis/               # Pre-generated HTML visualizations
│
├── custom_stopwords.txt        # User-defined stopwords
├── pyproject.toml              # Python dependencies
├── main.py                     # Backend entry point
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 18+ (or Bun)
- [uv](https://github.com/astral-sh/uv) package manager

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd topic-modeling-clustering

# Install Python dependencies
uv sync

# Install frontend dependencies
cd frontend && bun install && cd ..
```

### 2. Run Precomputation (Required First Time)

This trains LDA models for all topic counts (2-20), computes UMAP projections, and generates pyLDAvis visualizations. Takes approximately **30-90 minutes** depending on hardware.

```bash
# Full precomputation (k=2 to k=20)
uv run python scripts/precompute.py

# Quick test run (k=2 to k=4 only)
uv run python scripts/precompute.py --min-topics 2 --max-topics 4
```

### 3. Start the Application

**Terminal 1 - Backend API:**
```bash
uv run python main.py
# or
uv run uvicorn backend.app:app --reload
```
Backend runs at: http://localhost:8000

**Terminal 2 - Frontend Dev Server:**
```bash
cd frontend && bun run dev
```
Frontend runs at: http://localhost:5173

### 4. Open the Dashboard

Navigate to http://localhost:5173 in your browser.

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies |
| `uv run python scripts/precompute.py` | Run full precomputation pipeline |
| `uv run python scripts/precompute.py --min-topics 2 --max-topics 4` | Quick precompute for testing |
| `uv run python main.py` | Start backend server |
| `uv run uvicorn backend.app:app --reload` | Start backend with hot reload |
| `uv run pytest tests/ -v` | Run backend unit tests |
| `cd frontend && bun install` | Install frontend dependencies |
| `cd frontend && bun run dev` | Start frontend dev server |
| `cd frontend && bun run build` | Build frontend for production |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with cache status |
| `/api/status` | GET | Detailed precomputation status |
| `/api/topics/coherence` | GET | Coherence & perplexity scores for all k |
| `/api/topics/{n}/words` | GET | Top words per topic for k=n |
| `/api/topics/{n}/pyldavis` | GET | Interactive pyLDAvis HTML |
| `/api/topics/{n}/distribution` | GET | Document-topic matrix |
| `/api/clustering` | POST | Perform K-Means clustering |
| `/api/clustering/metrics/{n}` | GET | Silhouette/Inertia for all cluster counts |
| `/api/visualization/{n}` | GET | UMAP projections for k=n |
| `/api/visualization/clustered` | POST | Projections with cluster labels |

---

## Custom Stopwords

Add domain-specific stopwords to `custom_stopwords.txt`:

```text
# One word per line, # for comments
subject
wrote
article
edu
```

After editing, clear cached tokenized docs and re-run precomputation:
```bash
rm cache/models/tokenized_docs.pkl
uv run python scripts/precompute.py
```

---

## ML Engineer Impact & Use Cases

### Why This Matters

| Aspect | Impact |
|--------|--------|
| **Model Selection** | Compare coherence & perplexity across topic counts to find optimal k |
| **Interpretability** | Visualize topic-word distributions with pyLDAvis for stakeholder presentations |
| **Cluster Validation** | Use silhouette scores and elbow method to validate clustering quality |
| **Rapid Prototyping** | Precomputation + caching enables instant exploration without waiting for training |
| **Production Patterns** | Demonstrates caching strategies for expensive ML operations |

### Key ML Engineering Patterns Demonstrated

1. **Precomputation Strategy**
   - LDA training takes 1-5 min per model
   - Precompute all models (19 configs) upfront
   - Serve cached results for instant UI response

2. **Metric-Driven Model Selection**
   - **Coherence (c_v)**: Measures topic interpretability (higher = better)
   - **Perplexity**: Measures generalization (lower = better)
   - Both plotted together for informed selection

3. **Clustering Validation**
   - **Silhouette Score**: Cluster separation quality (-1 to 1)
   - **Inertia**: Within-cluster variance (for elbow detection)
   - Automatic elbow point detection using kneedle algorithm

4. **Dimensionality Reduction Pipeline**
   - Document-topic matrix (18K x k) → UMAP → 2D coordinates
   - Enables visual cluster exploration and outlier detection

5. **Text Preprocessing Best Practices**
   - Contraction expansion before tokenization
   - spaCy for lemmatization with POS-aware processing
   - Configurable stopword lists for domain adaptation

### Example Workflow for ML Engineers

```python
# 1. Analyze coherence scores to pick optimal topics
GET /api/topics/coherence
# Returns: {topic_counts: [2..20], coherence_scores: [...], perplexity_scores: [...]}

# 2. Examine topic words for interpretability
GET /api/topics/8/words
# Returns: {topics: [[{word, probability}, ...], ...]}

# 3. Cluster documents and evaluate quality
POST /api/clustering {n_topics: 8, n_clusters: 5}
# Returns: {labels: [...], silhouette: 0.42, inertia: 1234.5}

# 4. Get 2D projections for visualization
POST /api/visualization/clustered {n_topics: 8, n_clusters: 5}
# Returns: {projections: [[x, y], ...], cluster_labels: [...]}
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_api_topics.py -v

# Run with coverage
uv run pytest tests/ --cov=backend
```

---

## License

MIT License
