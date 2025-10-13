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
    subgraph Data["🗂️ Data Layer"]
        DS[("📊 IMDb Dataset<br/>movies_overview.csv<br/>movies_genres.csv")]
        DS --> LOAD[/"📥 Data Loader<br/>parse_genre_ids()<br/>load_and_clean_data()"/]
    end

    subgraph EDA["🔬 Exploratory Data Analysis"]
        LOAD --> EDA1["📈 Basic Stats<br/>• Genre frequency<br/>• Text length<br/>• Label counts"]
        LOAD --> EDA2["🔢 Multi-Label Metrics<br/>• Label Density<br/>• Cardinality<br/>• Gini coefficient"]
        LOAD --> EDA3["🔗 Dependency Analysis<br/>• Mutual Information<br/>• Phi Correlation<br/>• Co-occurrence"]
        LOAD --> EDA4["🖼️ Visualizations<br/>• Bar charts<br/>• Heatmaps<br/>• Distributions"]
        
        EDA1 & EDA2 & EDA3 & EDA4 --> EDAREP[/"📄 EDA Report<br/>eda_report.json<br/>+ plots/"/]
    end

    subgraph FE["⚙️ Feature Engineering"]
        EDAREP --> SPLIT["🔀 Stratified Split<br/>iterative_train_test_split()<br/>80% train / 20% val"]
        SPLIT --> TFIDF["📝 TF-IDF Vectorization<br/>• max_features=20k<br/>• ngrams=(1,2)<br/>• sublinear_tf=True"]
        TFIDF --> FEATS[("🎯 Features<br/>X_train, X_val<br/>(n × 20000)")]
    end

    subgraph ML["🤖 Model Training"]
        FEATS --> BR["🎲 Binary Relevance<br/>LogisticRegression × N<br/>• C=1.0<br/>• solver=liblinear"]
        FEATS --> MLKNN["🎯 ML-kNN<br/>Bayesian k-NN<br/>• k=10<br/>• s=1.0"]
        
        BR --> BRMET["📊 BR Metrics<br/>• Hamming: 0.18<br/>• F1-micro: 0.73<br/>• P@3: 0.78"]
        MLKNN --> MLKMET["📊 ML-kNN Metrics<br/>• Hamming: 0.21<br/>• F1-macro: 0.64<br/>• R@3: 0.71"]
        
        BRMET & MLKMET --> MLFLOW[("📦 MLflow<br/>Experiment Tracking<br/>Model Registry")]
    end

    subgraph SERVE["🚀 Model Serving"]
        MLFLOW --> API["🌐 FastAPI<br/>• /predict/transformation<br/>• /predict/adaptation<br/>• async endpoints"]
        API --> CACHE["💾 Redis Cache<br/>• Hash overviews<br/>• TTL=1h<br/>• 40% hit rate"]
        API --> LB["⚖️ Load Balancer<br/>Round-robin<br/>Health checks"]
    end

    subgraph MONITOR["📊 Monitoring & Observability"]
        API --> PROM["📈 Prometheus<br/>• Request rate<br/>• Inference time<br/>• Confidence scores"]
        PROM --> GRAF["📊 Grafana<br/>• Real-time dashboards<br/>• Alerting<br/>• Visualization"]
        PROM --> ALERT["🚨 Alertmanager<br/>• High error rate<br/>• Slow inference<br/>• Low confidence"]
    end

    subgraph DEPLOY["🐳 Deployment"]
        LB --> DOCKER["🐳 Docker<br/>• Multi-stage build<br/>• Non-root user<br/>• Health checks"]
        DOCKER --> K8S["☸️ Kubernetes<br/>• 3-10 replicas<br/>• HPA (CPU/Memory)<br/>• Rolling updates"]
    end

    subgraph FEEDBACK["🔄 Continuous Improvement"]
        GRAF --> DRIFT["📉 Drift Detection<br/>• Compare metrics<br/>• Track confidence<br/>• Monitor data dist"]
        DRIFT --> RETRAIN{"🔄 Retrain?<br/>IF perf drops >5%"}
        RETRAIN -->|Yes| LOAD
        RETRAIN -->|No| MONITOR
        
        ALERT --> ONCALL["📞 On-Call Engineer<br/>• Investigate<br/>• Rollback if needed"]
    end

    subgraph FUTURE["🚀 Future: LLM Pipeline"]
        style FUTURE fill:#e1f5ff,stroke:#01579b,stroke-width:3px
        
        LLM["🤖 LLM Orchestration<br/>LangChain + GPT-4"]
        LLM --> AUG["📝 Data Augmentation<br/>• Paraphrase<br/>• Generate test cases"]
        LLM --> ACTIVE["🎯 Active Learning<br/>• Flag low confidence<br/>• Human-in-the-loop"]
        AUG & ACTIVE --> RETRAIN
    end

    style DS fill:#fff3e0,stroke:#e65100
    style MLFLOW fill:#e8f5e9,stroke:#2e7d32
    style API fill:#e3f2fd,stroke:#1565c0
    style PROM fill:#fce4ec,stroke:#c2185b
    style DOCKER fill:#f3e5f5,stroke:#6a1b9a
