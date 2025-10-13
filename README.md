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
```
## Monitoring & Observability
Why Monitoring Matters for ML Systems
ML models degrade over time:

Data drift: User behavior changes, new movie styles emerge
Concept drift: Genre definitions evolve (e.g., "dark comedy")
Performance drift: Model gets stale, needs retraining

Without monitoring, you won't know your model is broken until users complain.
Prometheus Metrics
We expose 4 categories of metrics:
1. Request Metrics:
# Total requests
http_requests_total{method="POST", endpoint="/predict/transformation", status="200"}

# Request latency (histogram)
http_request_duration_seconds{endpoint="/predict/transformation"}

2. Model Inference Metrics:
# Inference time per model
model_inference_duration_seconds{model_type="transformation"} 
model_inference_duration_seconds{model_type="adaptation"}

3. Prediction Confidence
# Confidence scores per genre
model_prediction_confidence{model_type="transformation", genre="Action"}
model_prediction_confidence{model_type="transformation", genre="Sci-Fi"}

Why it matters:
Sudden drop in confidence → Model uncertainty (investigate)
High confidence on rare genres → Potential false positives

4. Request Volume per Model
# How many predictions per model
model_requests_total{model_type="transformation"}
model_requests_total{model_type="adaptation"}

Why it matters: Load balancing, A/B testing analysis

## Grafana Dashboard
Our dashboard has 4 panels:

Request Rate (RPS)

Time series of requests/second
Detect traffic spikes, DDoS, or outages


Inference Time by Model

Compare Binary Relevance vs. ML-kNN latency
Identify slow models


Request Volume by Model

Stacked area chart showing model usage
Useful for A/B testing


Average Prediction Confidence

Single stat showing overall model confidence
Drops below 0.5 → model struggling


Installation & Setup
Prerequisites

Python 3.10+
Docker & Docker Compose (for containerized deployment)
4GB RAM minimum (8GB recommended)
2GB free disk space

Quick Start
### 1. Clone repository
git clone <repo>
cd <repo>

### 2. Create virtual environment
Recommended: uv
uv init .
uv venv && .venv\Scripts\activate
### 3. Install dependencies
uv pip install  "fastapi[all]" pandas numpy scikit-learn scikit-multilearn mlflow matplotlib seaborn joblib prometheus-client

### 4. Download dataset (place in IMDb-dataset/)
- movies_overview.csv
- movies_genres.csv

### 5. Run EDA
python -m src.run_eda

### scikit-multilearn bug
in .venv/Lib/skmultilearn/adapt/mlknn.py, replace this line: self.knn_ = NearestNeighbors(self.k).fit(X) with this: self.knn_ = NearestNeighbors(n_neighbors = self.k).fit(X)

### 6. Train models
python -m src.run_trainer

### 7. Start API
python serve.py

### 8. Test prediction
curl -X POST "http://localhost:8000/predict/transformation" \
  -H "Content-Type: application/json" \
  -d '{"overview": "A team of astronauts travels through a wormhole in space."}'


## Docker Deployment
# Build and start all services
docker-compose up -d

# Access services
# API:        http://localhost:8000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000 (admin/admin)

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

## Future Enhancements
### 1. Real-Time Test Data Pipeline with LLM Orchestration
Problem: Models degrade as real-world data changes. We need continuous evaluation on fresh, realistic test data.
Benefits
Continuous Evaluation: New test data generated daily
Diverse Coverage: LLM can generate edge cases humans miss
Cost-Effective: No manual labeling needed
Drift Detection: Early warning when model performance degrades

Challenges

LLM Bias: Generated plots may not reflect real movies
Quality Control: Need human validation for ground truth
Cost: API calls to GPT-4/Claude can be expensive
Latency: Test generation is slow (not real-time)

Mitigation: Hybrid approach with 90% synthetic + 10% real user data (labeled via active learning)

### 2. Scalability & Resilience Improvements
#### Horizontal Scaling with Kubernetes
Deploy API with auto-scaling (3-10 replicas) based on CPU/memory utilization. Use HorizontalPodAutoscaler to handle traffic spikes automatically. Add liveness and readiness probes for self-healing. Impact: Handle 10x traffic without manual intervention.
#### Caching Layer (Redis)
Cache predictions based on overview hash (SHA256) with 1-hour TTL. Expected: 40% cache hit rate for repeated queries, reduces inference load by 40%, lowers latency from 20ms → 2ms for cached requests.
#### Message Queue for Async Predictions (Celery)
Add Celery workers with Redis broker for batch processing. Users submit prediction tasks, receive task IDs, poll for results. Use case: Process 10,000 movies overnight without blocking API.
#### Circuit Breaker Pattern
Implement circuit breaker (fail-fast after 5 consecutive failures, recover after 60s). 
Prevents cascading failures when model service is down. Impact: API stays responsive even when ML service degrades.

### 4. Model Versioning & A/B Testing

Shadow Mode: Run new model alongside production, log predictions without serving
Canary Deployment: Route 5% traffic to new model, monitor metrics, full rollout if successful
Champion/Challenger: Always keep 2 models deployed, compare weekly, promote better model

### 5. Enhanced Monitoring

Data Quality Monitoring: Track input distribution drift (overview length, vocabulary changes)
Prediction Distribution: Monitor genre prediction frequency over time
Business Metrics: Track user engagement with predicted genres (if available)

## Deep Learning Extensions
Why Deep Learning for Genre Prediction?
Current TF-IDF + Linear models have limitations:

No Semantic Understanding: "happy" ≠ "joyful" (different words)
Context-Free: "bank" (river) vs "bank" (financial) treated the same
Fixed Vocabulary: Can't handle new words (e.g., "metaverse")
No Transfer Learning: Trained from scratch, ignores pre-trained knowledge

Deep learning solves these via:

Embeddings: Semantically similar words have similar vectors
Contextualization: Transformers understand word meaning from context
Transfer Learning: Fine-tune pre-trained models (BERT, GPT)

## Proposed Architecture 1: BERT for Multi-Label Classification
How it would work: Fine-tune bert-base-uncased (12 transformer layers) on movie overviews by feeding the [CLS] token representation through a classification head (768→256→num_genres with sigmoid activation). The model would learn contextual embeddings where "dark comedy" is understood differently from "dark thriller" and synonyms like "space"/"cosmos"/"galaxy" map to similar regions. Training takes ~3 epochs with learning rate 2e-5 and requires GPU (2-4 hours on single V100).
Why it's better: BERT captures semantic meaning that TF-IDF misses—it understands that "heist" relates to "Crime" even if that exact word wasn't in training. Expected improvements: +11% F1-micro (0.73→0.81), +18% F1-macro (0.61→0.72), especially helping rare genres like Documentary. Trade-off: inference latency increases from 20ms to ~80ms, and model size grows from 57MB to 440MB.

## Proposed Architecture 2: Hierarchical Attention Network
How it would work: Split each overview into sentences, encode each sentence with a Bi-LSTM + attention (word-level), then encode the sequence of sentence vectors with another Bi-LSTM + attention (sentence-level) to get a document representation. For example, "A thief infiltrates dreams." → sentence vector → combined with other sentences → final genre prediction. This two-level hierarchy explicitly models document structure.
Why it's better: Interpretability—we can visualize which sentences contribute to which genre (e.g., "space station" sentence → Sci-Fi). More efficient than BERT (~30ms inference vs 80ms) while still capturing context. Particularly useful when overviews have clear narrative structure: introduction sentence, conflict sentence, climax sentence each may hint at different genres. Expected +6-8% F1 improvement with lower computational cost.

## Proposed Architecture 3: Multi-Task Learning
How it would work: Share a BERT/LSTM encoder across multiple prediction tasks—genre (multi-label), rating (regression), release era (1920s/1950s/2000s multi-class), and budget tier (low/medium/high). The shared encoder learns representations that help all tasks: high ratings correlate with Drama, big budgets with Action. Training jointly with loss = 0.4·L_genre + 0.2·L_rating + 0.2·L_era + 0.2·L_budget.
Why it's better: Regularization through auxiliary tasks prevents overfitting—the model can't just memorize genre-specific keywords. Data efficiency—if we have rating/era/budget labels for only 50% of movies, we still leverage that information. Expected +5% F1 improvement plus better generalization to unseen movie types. Bonus: single API call returns genre + rating prediction together.

## Proposed Architecture 4: Label Attention Network (LAN)
How it would work: Use a transformer encoder on the overview, then apply genre-specific attention—for each genre, compute attention weights over all tokens to find relevant parts. "Action" head attends to "chase," "explosion"; "Romance" head attends to "love," "relationship." Each genre gets its own attention-weighted document representation → binary classifier. This is like having 20 specialized "readers" each looking for their genre's keywords.
Why it's better: Handles label correlations naturally—if "Action" and "Adventure" heads both attend to "jungle expedition," the model learns they co-occur. More parameter-efficient than Binary Relevance because the transformer encoder is shared. Expected +7-9% F1-macro particularly helping rare genres by giving them dedicated attention mechanisms. Works well with our high co-occurrence (Action-Adventure NMI=0.42) findings from EDA.


## Real-Time Pipeline with LLM Orchestration
### Problem Statement
ML models degrade over time due to data drift (user behavior changes) and concept drift (genre definitions evolve). Traditional pipelines rely on batch retraining schedules, but we need continuous evaluation on fresh, realistic test data to detect degradation early.
### Key Components
1. LLM Test Generator (LangChain + GPT-4)

Generates diverse movie plots for any genre combination
Prompt engineering ensures realistic overviews (50-150 words)
Temperature = 0.7 for creative variety
Cost: ~$0.002 per test case (affordable at scale)

2. Test-Time Augmentation

Generate 5 paraphrases of each overview using LLM
Predict on all 6 variants (original + 5 paraphrases)
Ensemble via majority voting
Result: Reduces prediction variance by 30%, improves confidence calibration

3. Active Learning Loop

Flag predictions with confidence < 0.7 for human review
Humans label ~100 cases/day (10 min/day effort)
Retrain weekly with new labeled data
Result: Continuous improvement without expensive labeling campaigns

4. Drift Detection Dashboard

Track F1, Precision@3, Recall@3 over time (rolling 7-day window)
Alert if any metric drops > 3% from baseline
Visualize per-genre performance to identify specific drift
Result: Early warning system (detect issues in days, not months)

### Benefits

Early Drift Detection: Identify performance degradation in 24-48 hours vs. weeks
Cost-Effective: $2/day for 1000 synthetic test cases vs. $500/day for manual labeling
Diverse Coverage: LLM generates edge cases humans wouldn't think of (e.g., "sci-fi western fusion")
Continuous Improvement: Models get better over time with minimal human effort

## Pipeline Flow Summary
The diagram above shows the actual implemented system with these phases:

Data Ingestion → Load CSVs and clean data
EDA → Compute 14 multi-label metrics + visualizations
Training → Train Binary Relevance & ML-kNN models
Serving → FastAPI with 2 prediction endpoints
Monitoring → Prometheus + Grafana stack
Deployment → Docker Compose with 3 services
