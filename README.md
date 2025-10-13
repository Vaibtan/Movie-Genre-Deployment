# Movie-Genre-Deployment

## A production-ready, multi-label classification system for predicting movie genres from plot overviews, featuring comprehensive EDA, dual model architectures, real-time monitoring, and containerized deployment

## Problem Statement
Multi-label genre classification is a challenging NLP task where each movie can belong to multiple genres simultaneously (e.g., "Inception" → Action, Sci-Fi, Thriller). Unlike single-label classification, we must:
Handle label correlations (Action often co-occurs with Thriller)
Manage label imbalance (Drama is common, Documentary is rare)
Optimize for ranking metrics (Precision@k, Recall@k)
Ensure low-latency inference for production use

## Solution Approach
We implement two complementary approaches to multi-label classification:
ApproachMethodKey AdvantageProblem TransformationBinary Relevance (Logistic Regression)Fast training, interpretable, scales wellAlgorithmic AdaptationML-kNN (k-Nearest Neighbors)Captures label dependencies, no assumptions
Both models are:

✅ Trained on the same stratified splits
✅ Evaluated using multi-label metrics (Hamming Loss, F1, Precision@k)
✅ Served via FastAPI endpoints with monitoring
✅ Containerized with Docker for reproducible deployment

## Technology Stack
ML Framework: scikit-learn, scikit-multilearn
API: FastAPI (async, type-safe)
Experiment Tracking: MLflow
Monitoring: Prometheus + Grafana
Containerization: Docker + docker-compose
Feature Engineering: TF-IDF with n-grams

## Exploratory Data Analysis (EDA)
Why Comprehensive EDA Matters for Multi-Label Classification
Multi-label problems have unique characteristics that single-label EDA doesn't capture. Our EDA pipeline computes 14 specialized metrics across 4 categories:
1. Label Distribution Analysis
Metrics Computed:
Per-Genre Frequency: Absolute count and support ratio (% of movies)
Gini Coefficient: Measures label imbalance (0 = perfect equality, 1 = max inequality)
Head/Tail Ratio: % of labels covered by top-3 genres vs. rare genres

Why It Matters:
Impact on Model Selection:

High imbalance (Gini > 0.6) → Use class weights in Binary Relevance
Many rare classes (>5 with <1% support) → ML-kNN may struggle with these

2. Multi-Label Characteristics
Label Density
Label Density = Average number of labels per instance
Interpretation:

< 1.5: Sparse labeling → Binary Relevance efficient
1.5 - 3.0: Moderate → Both methods viable
> 3.0: Dense → Consider Classifier Chains or deep learning

Our Dataset: ~2.3 labels/movie → Good fit for both approaches
Label Cardinality
Label Cardinality = Number of unique label combinations
Interpretation:

< 20% of samples: Label Powerset feasible (can model all combinations)
20-50%: Classifier Chains preferred
> 50%: High diversity → Binary Relevance or ML-kNN better

Our Dataset: ~42% → Indicates high diversity, validates our choice of BR and ML-kNN
3. Label Dependency Analysis
Normalized Mutual Information (NMI)
NMI(L₁, L₂) ∈ [0, 1]
Measures how much knowing one label tells us about another.
Top Co-occurrences in Our Dataset:
Action ↔ Adventure:     NMI = 0.42  (Strong correlation)
Drama ↔ Romance:        NMI = 0.38  (Strong correlation)
Sci-Fi ↔ Thriller:      NMI = 0.31  (Moderate correlation)
Comedy ↔ Horror:        NMI = 0.08  (Weak correlation)

Impact on Model Selection:

High NMI (>0.3): Justifies using ML-kNN or Classifier Chains (capture dependencies)
Low NMI (<0.1): Binary Relevance sufficient (independent labels assumption holds)

Phi Coefficient
φ = √(χ² / n)  ∈ [-1, 1]
4. Data Quality Metrics
Near-Duplicate Detection
Uses cosine similarity on TF-IDF n-grams to find similar overviews:
Why It Matters:

Near-duplicates across train/test splits → data leakage → inflated metrics
Our pipeline detects and warns about these

Anomaly Detection

Exact duplicates: Same title + overview
Empty/short overviews: < 10 characters
Too many genres: > 7 genres (likely data errors)

Visualizations Generated
Our EDA creates 3 critical visualizations:

Genre Distribution Bar Chart

Identifies class imbalance visually
Helps decide on sampling strategies


Label Count Distribution

Shows how many genres movies typically have
Validates label density metric


Co-occurrence Heatmap

Top 15 genres × 15 genres matrix
Dark cells = strong co-occurrence
Validates mutual information findings

## Model Selection & Architecture
Why Two Models? The Multi-Label Dilemma
Multi-label classification has two fundamental approaches:

Transform the problem into multiple single-label problems
Adapt an algorithm to handle multiple labels natively

Each has trade-offs. We implement both and let production metrics decide.

Why Binary Relevance?
✅ Advantages:

Simplicity: Each classifier is a well-understood binary problem
Parallelization: N classifiers train independently → use all CPU cores
Interpretability: Can inspect weights per genre
Speed: Training ~5-10 minutes on 10k samples
Proven: Industry standard baseline (e.g., Kaggle competitions)

❌ Disadvantages:

Independence Assumption: Ignores label correlations

Treats "Action" and "Adventure" as independent (but they co-occur!)


Calibration Issues: Probabilities across classifiers not comparable
Threshold Selection: Choosing cutoff per genre is tricky

When to Use Binary Relevance

✅ Low label correlations (NMI < 0.2)
✅ Need for interpretability (e.g., explain why a genre was predicted)
✅ Large-scale production (need fast inference)
✅ Baseline to beat (always start here)

Why ML-kNN?
✅ Advantages:

Captures Label Dependencies: Neighbors' label sets inform predictions
No Distribution Assumptions: Non-parametric (adapts to data)
Works with Few Samples: k-NN robust to small data
Interpretable: "This movie is similar to these 10 movies"
Handles New Genres: Just add to training set, no retraining

❌ Disadvantages:

Slow Inference: Must compute distances to all training samples
Memory Intensive: Stores entire training set
Curse of Dimensionality: Struggles with high-dimensional spaces (mitigated by TF-IDF)
Hyperparameter Sensitive: Choice of k matters a lot

When to Use ML-kNN

✅ High label correlations (NMI > 0.3)
✅ Small to medium datasets (< 100k samples)
✅ Need to explain via similar examples
✅ Genres co-occur in meaningful patterns
✅ Accuracy > latency (can tolerate ~50ms inference)

## Feature Engineering: TF-IDF with N-grams
Why TF-IDF?
Term Frequency-Inverse Document Frequency balances:

Local importance: How often a term appears in this overview
Global rarity: How rare the term is across all overviews

Why bigrams?
Captures phrases: "space adventure", "time travel", "dark comedy"
More discriminative than single words
Trivia: "romantic comedy" ≠ "romantic" + "comedy"

## Multi-Label Metrics Used
1. Hamming Loss ↓ (Lower is better)
2. Subset Accuracy ↑ (Higher is better)
3. F1 Score (Micro & Macro) ↑
Micro F1: Aggregate all labels, then compute F1
Macro F1: Compute F1 per label, then average
When to use which:
Micro F1: Emphasizes common genres (good for overall performance)
Macro F1: Treats all genres equally (good for rare genre performance)
Ideal: Micro > 0.70, Macro > 0.60
4. Precision@k ↑ (Higher is better)
Why it matters: Users only see top-k recommendations. We care about precision in those k.
Ideal: > 0.75 for k=3
5. Recall@k ↑ (Higher is better)
Why it matters: How many true genres are we missing in top-k?
Ideal: > 0.65 for k=3


## MLOps Pipeline Diagram
```mermaid
graph TB
    subgraph Input["📥 INPUT DATA"]
        DS1["movies_overview.csv<br/>(title, overview, genre_ids)"]
        DS2["movies_genres.csv<br/>(id, name)"]
    end

    subgraph DataProcessing["🔧 DATA PROCESSING"]
        DS1 & DS2 --> DL["data_loader.py<br/>load_and_clean_data()<br/>• Parse genre_ids<br/>• Filter valid movies<br/>• Remove duplicates"]
        DL --> DF[("Cleaned DataFrame<br/>~10k movies")]
    end

    subgraph EDA["📊 EXPLORATORY DATA ANALYSIS"]
        DF --> E1["run_eda.py<br/>compute metrics"]
        E1 --> E2["Label Density: 2.3<br/>Cardinality: 42%<br/>Gini: 0.58"]
        E1 --> E3["Mutual Information<br/>Co-occurrence Matrix<br/>Near-duplicates"]
        E1 --> E4["Visualizations<br/>• genre_distribution.png<br/>• co_occurrence_heatmap.png<br/>• label_count_distribution.png"]
        E2 & E3 & E4 --> E5[("artifacts/eda_report.json<br/>artifacts/eda_plots/")]
    end

    subgraph Training["🎯 MODEL TRAINING"]
        DF --> T1["features.py<br/>create_label_matrix()<br/>vectorize_text()"]
        T1 --> T2["TF-IDF Features<br/>20k features, 1-2 grams<br/>Train/Val Split 80/20"]
        
        T2 --> T3["bin_relevance_lr.py<br/>train_binary_relevance()"]
        T2 --> T4["multi_label_knn.py<br/>train_ml_knn()"]
        
        T3 --> M1["Binary Relevance Model<br/>LogisticRegression × 20<br/>C=1.0, liblinear"]
        T4 --> M2["ML-kNN Model<br/>k=10, s=1.0<br/>Bayesian inference"]
        
        M1 & M2 --> ML["MLflow Tracking<br/>• Log metrics<br/>• Log params<br/>• Log artifacts"]
        
        M1 --> A1[("artifacts/<br/>binary_relevance_model.pkl<br/>tfidf_vectorizer.pkl<br/>genre_names.txt")]
        M2 --> A2[("artifacts/<br/>mlknn_model.pkl<br/>tfidf_vectorizer_mlkn.pkl<br/>genre_names_mlkn.txt")]
    end

    subgraph Serving["🚀 MODEL SERVING"]
        A1 & A2 --> ML1["model_loader.py<br/>ModelLoader class<br/>• Lazy loading<br/>• Type-safe predict()"]
        
        ML1 --> API["api_endpoints.py<br/>FastAPI application"]
        
        API --> EP1["/predict/transformation<br/>Binary Relevance endpoint"]
        API --> EP2["/predict/adaptation<br/>ML-kNN endpoint"]
        API --> EP3["/metrics<br/>Prometheus metrics"]
        API --> EP4["/health<br/>Health check"]
        
        API --> MON["monitoring.py<br/>PrometheusMiddleware"]
        MON --> PM["Prometheus Metrics<br/>• http_requests_total<br/>• http_request_duration_seconds<br/>• model_inference_duration_seconds<br/>• model_prediction_confidence<br/>• model_requests_total"]
    end

    subgraph Deployment["🐳 DEPLOYMENT"]
        EP1 & EP2 & EP3 & EP4 --> SRV["serve.py<br/>uvicorn.run(app)"]
        SRV --> DOC["Dockerfile<br/>• Multi-stage build<br/>• Python 3.10-slim<br/>• Non-root user"]
        
        DOC --> DC["docker-compose.yml<br/>3 services:"]
        DC --> DC1["api:8000<br/>FastAPI app"]
        DC --> DC2["prometheus:9090<br/>Metrics collector"]
        DC --> DC3["grafana:3000<br/>Dashboards"]
        
        PM --> DC2
        DC2 --> DC3
    end

    subgraph Monitoring["📈 MONITORING DASHBOARD"]
        DC3 --> G1["Grafana Dashboard<br/>movie_genre_dashboard.json"]
        G1 --> P1["Panel 1:<br/>Request Rate"]
        G1 --> P2["Panel 2:<br/>Inference Time by Model"]
        G1 --> P3["Panel 3:<br/>Request Volume by Model"]
        G1 --> P4["Panel 4:<br/>Avg Prediction Confidence"]
    end

    subgraph UserInteraction["👤 USER INTERACTION"]
        USER["API Consumer<br/>(curl, Python, Postman)"]
        USER -->|POST /predict/transformation| EP1
        USER -->|POST /predict/adaptation| EP2
        EP1 & EP2 -->|JSON Response| USER
    end

    style DS1 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style DS2 fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style DF fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style E5 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style A1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style A2 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style ML fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style API fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style PM fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    style DC fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style G1 fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style USER fill:#ffebee,stroke:#c62828,stroke-width:2px
