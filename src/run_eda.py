import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_and_clean_data
from schema import (AnomalyReport, CoOccurrencePair, EDAResults, GenreStats,
                    ImbalanceMetrics, LabelDensityMetrics, MutualInfoPair,
                    TextStats)
from storage_manager import StorageManager


# Compute Gini coefficient of inequality (0 = perfect equality, 1 = max inequality)
def gini_coefficient(array: List[int]) -> float:
    array = sorted(array)
    n: int = len(array)
    if n == 0 or sum(array) == 0: return 0.0
    indices: List[int] = list(range(1, n + 1))
    numerator: int = 2 * sum(i * val for i, val in zip(indices, array))
    denominator: int = n * sum(array)
    return numerator / denominator - (n + 1) / n

def compute_label_density_cardinality(movies_df: pd.DataFrame) -> Tuple[float, int, Dict[Tuple[int, ...], int]]:
    """
    Compute label density, cardinality, and label combination frequencies.
    Label Density: Average number of labels per instance
    Label Cardinality: Number of unique label combinations
    Returns: (density, cardinality, combination_counts)
    """
    label_counts: npt.NDArray[np.int_] = movies_df["genre_ids"].apply(len).values
    density: float = float(np.mean(label_counts))
    # Convert to tuples for hashability
    label_combinations: List[Tuple[int, ...]] = [tuple(sorted(genres)) for genres in movies_df["genre_ids"]]
    combination_counter: Counter[Tuple[int, ...]] = Counter(label_combinations)
    cardinality: int = len(combination_counter)
    return density, cardinality, dict(combination_counter)


def compute_mutual_information(movies_df: pd.DataFrame, genres_df: pd.DataFrame, \
    top_k: int = 10) -> List[Tuple[str, str, float]]:
    """
    Compute normalized mutual information between all genre pairs.
    Args: top_k: Number of top correlations to return
    Returns: List of (genre_a, genre_b, nmi_score)
    """
    all_genre_ids: Set[int] = set(genres_df["id"])
    genre_vectors: Dict[int, npt.NDArray[np.int_]] = {}
    for gid in all_genre_ids:
        genre_vectors[gid] = movies_df["genre_ids"].apply(lambda x: 1 if gid in x else 0).values
    nmi_scores: List[Tuple[int, int, float]] = []
    genre_ids: List[int] = sorted(all_genre_ids)
    for i, gid1 in enumerate(genre_ids):
        for gid2 in genre_ids[i + 1:]:
            nmi: float = normalized_mutual_info_score(genre_vectors[gid1], genre_vectors[gid2])
            nmi_scores.append((gid1, gid2, nmi))
    nmi_scores.sort(key = lambda x: x[2], reverse = True)    
    # Map to genre names
    genre_map: Dict[int, str] = dict(zip(genres_df["id"], genres_df["name"]))
    top_nmi: List[Tuple[str, str, float]] = [(genre_map[g1], genre_map[g2], score) \
        for g1, g2, score in nmi_scores[:top_k]]
    return top_nmi


# Compute Phi coefficient (correlation) between two binary genre variables (between -1 and 1)
def compute_phi_coefficient(movies_df: pd.DataFrame, genre1_id: int, genre2_id: int) -> float:
    g1_present: npt.NDArray[np.bool_] = movies_df["genre_ids"].apply(lambda x: genre1_id in x).values
    g2_present: npt.NDArray[np.bool_] = movies_df["genre_ids"].apply(lambda x: genre2_id in x).values
    # Create contingency table
    both: int = int(np.sum(g1_present & g2_present))
    g1_only: int = int(np.sum(g1_present & ~g2_present))
    g2_only: int = int(np.sum(~g1_present & g2_present))
    neither: int = int(np.sum(~g1_present & ~g2_present))
    contingency: npt.NDArray[np.int_] = np.array([[both, g1_only], [g2_only, neither]])
    # Compute phi coefficient
    chi2_stat: float
    chi2_stat, _, _, _ = chi2_contingency(contingency)
    n: int = len(movies_df)
    phi: float = np.sqrt(chi2_stat / n) 
    return float(phi)


def detect_near_duplicates(movies_df: pd.DataFrame, sim_threshold: float = 0.9) -> List[Tuple[int, int, float]]:
    """
    Detect near-duplicate movie overviews using Jaccard similarity.
    Args: sim_threshold: Minimum Jaccard similarity to flag 
    Returns: List of (idx1, idx2, similarity)
    """
    if len(movies_df) > 5000: sample_df: pd.DataFrame = movies_df.sample(n=5000, random_state=42)
    else: sample_df = movies_df
    vectorizer = TfidfVectorizer(max_features = 1000, ngram_range = (2, 3))
    tfidf_matrix = vectorizer.fit_transform(sample_df["overview"])
    # Compute cosine similarity
    similarity_matrix: npt.NDArray[np.float64] = cosine_similarity(tfidf_matrix)
    # Find high similarity pairs (excluding diagonal)
    duplicates: List[Tuple[int, int, float]] = []
    n: int = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] >= sim_threshold:
                duplicates.append((i, j, float(similarity_matrix[i, j])))
    return duplicates

# Create comprehensive EDA visualizations
def create_visualizations(movies_df: pd.DataFrame, genres_df: pd.DataFrame, \
    genre_stats: List[GenreStats], output_dir: Path) -> None:
    output_dir.mkdir(parents = True, exist_ok = True)
    plt.figure(figsize = (14, 6))
    genre_names: List[str] = [g.genre_name for g in genre_stats]
    counts: List[int] = [g.count for g in genre_stats]
    plt.bar(range(len(genre_names)), counts, color = 'steelblue')
    plt.xticks(range(len(genre_names)), genre_names, rotation = 45, ha = 'right')
    plt.ylabel('Frequency'); plt.title('Genre Distribution')
    plt.tight_layout(); plt.savefig(output_dir / "genre_distribution.png", dpi = 150)
    plt.close()
    # 2. Label count distribution
    plt.figure(figsize = (10, 6))
    label_counts: npt.NDArray[np.int_] = movies_df["genre_ids"].apply(len).values
    plt.hist(label_counts, bins = range(1, max(label_counts) + 2), color = 'coral', edgecolor = 'black', alpha = 0.7)
    plt.xlabel('Number of Genres per Movie'); plt.ylabel('Frequency')
    plt.title('Distribution of Label Counts'); plt.tight_layout()
    plt.savefig(output_dir / "label_count_distribution.png", dpi = 150)
    plt.close()
    # 3. Co-occurrence heatmap (top 15 genres)
    top_genres: List[int] = [genres_df[genres_df["name"] == \
        g.genre_name].iloc[0]["id"] for g in genre_stats[:15]]
    co_occurrence: npt.NDArray[np.int_] = np.zeros((len(top_genres), len(top_genres)), dtype=int)
    for genres in movies_df["genre_ids"]:
        genre_list: List[int] = [g for g in genres if g in top_genres]
        for i, g1 in enumerate(top_genres):
            for j, g2 in enumerate(top_genres):
                if g1 in genre_list and g2 in genre_list: co_occurrence[i, j] += 1
    
    plt.figure(figsize = (12, 10))
    top_genre_names: List[str] = [genres_df[genres_df["id"] == gid].iloc[0]["name"] \
        for gid in top_genres]
    sns.heatmap(co_occurrence, xticklabels = top_genre_names, yticklabels = top_genre_names, \
        annot = True, fmt = 'd', cmap = 'YlOrRd', cbar_kws = {'label': 'Co-occurrence Count'})
    plt.title('Genre Co-occurrence Heatmap (Top 15 Genres)')
    plt.tight_layout()
    plt.savefig(output_dir / "co_occurrence_heatmap.png", dpi = 150)
    plt.close()
    print(f"✅ Visualizations saved to {output_dir}")

def run_eda(movies_df: pd.DataFrame, genres_df: pd.DataFrame, output_path: Optional[Path] = \
    None, create_plots: bool = True, storage_manager: Optional[StorageManager] = None) -> EDAResults:
    print("📊 Running Exploratory Data Analysis...")
    # Basic counts
    n_movies, n_genres = len(movies_df), len(genres_df)
    movies_df["char_len"] = movies_df["overview"].str.len()
    avg_chars: float = float(movies_df["char_len"].mean())
    med_chars: float = float(movies_df["char_len"].median())
    p95_chars: float = float(movies_df["char_len"].quantile(0.95))
    # Label density and cardinality
    density, cardinality, combination_counts = compute_label_density_cardinality(movies_df)
    print(f"Label Density: {density:.3f}")
    print(f"Label Cardinality: {cardinality}")
    # Genre frequency analysis
    exploded: pd.DataFrame = movies_df.explode("genre_ids")
    genre_counts: pd.DataFrame = exploded["genre_ids"].value_counts().reset_index()
    genre_counts.columns = ["genre_id", "count"]
    genre_counts_named: pd.DataFrame = genre_counts.merge(genres_df[["id", "name"]], \
        left_on = "genre_id", right_on = "id", how = "inner")[["name", "count"]].rename(columns={"name": "genre_name"})
    total_label_instances: int = int(genre_counts_named["count"].sum())
    genre_stats: List[GenreStats] = [GenreStats(genre_name = row["genre_name"], \
        count = int(row["count"]), support_ratio = 100.0 * row["count"] / n_movies) \
            for _, row in genre_counts_named.iterrows()]
    # Imbalance metrics
    counts_list: List[int] = [g.count for g in genre_stats]
    gini: float = gini_coefficient(counts_list)
    head_ratio: float = (sum(g.count for g in genre_stats[:3]) / total_label_instances \
        if total_label_instances > 0 else 0.0)
    tail_count: int = sum(1 for g in genre_stats if g.support_ratio < 1.0)
    # Labels per movie
    movies_df["n_genres"] = movies_df["genre_ids"].apply(len)
    avg_genres: float = float(movies_df["n_genres"].mean())
    max_genres: int = int(movies_df["n_genres"].max())
    # Co-occurrence analysis
    pair_counter: Counter[Tuple[int, int]] = Counter()
    for genres in movies_df["genre_ids"]:
        if len(genres) > 1:
            for pair in combinations(sorted(genres), 2): pair_counter[pair] += 1
    top_pairs: List[Tuple[Tuple[int, int], int]] = pair_counter.most_common(10)
    genre_map: Dict[int, str] = dict(zip(genres_df["id"], genres_df["name"]))
    co_occurrences: List[CoOccurrencePair] = [CoOccurrencePair(genre_a = genre_map[g1], \
        genre_b = genre_map[g2], count = cnt) for (g1, g2), cnt in top_pairs]
    # Mutual information analysis
    print("Computing mutual information between genres...")
    top_nmi: List[Tuple[str, str, float]] = compute_mutual_information(movies_df, genres_df, top_k = 10)
    mutual_info_pairs: List[MutualInfoPair] = [MutualInfoPair(genre_a = g1, genre_b = g2, \
        nmi_score = score) for g1, g2, score in top_nmi]
    # Near-duplicate detection
    print("Detecting near-duplicate overviews...")
    duplicates: List[Tuple[int, int, float]] = detect_near_duplicates(movies_df, 0.9)
    # Anomalies
    dup_count: int = int(movies_df.duplicated(subset=["title", "overview"]).sum())
    short_count: int = int((movies_df["overview"].str.len() < 10).sum())
    too_many_genres: int = int((movies_df["n_genres"] > 7).sum())
    # Warnings
    warnings: List[str] = []
    if head_ratio > 0.6:
        warnings.append(f"High label imbalance: top 3 genres cover {head_ratio:.1%} of all labels.")
    if tail_count > 5:
        warnings.append(f"Many rare genres ({tail_count} with <1% support) may hurt recall.")
    if dup_count > 0:
        warnings.append(f"Found {dup_count} exact duplicate (title + overview) entries.")
    if len(duplicates) > 10:
        warnings.append(f"Found {len(duplicates)} near-duplicate pairs (>90% similarity).")
    if short_count > 50:
        warnings.append(f"{short_count} movies have very short overviews (<10 chars).")
    if cardinality > 0.5 * n_movies:
        warnings.append(
            f"High label cardinality ({cardinality}/{n_movies} = {cardinality/n_movies:.1%}) "
            "suggests Label Powerset may be infeasible; consider Classifier Chains."
        )    
    report = EDAResults(
        n_movies = n_movies,
        n_genres = n_genres,
        avg_genres_per_movie = avg_genres,
        max_genres_in_one_movie = max_genres,
        label_density = density,
        label_cardinality = cardinality,
        genre_distribution = genre_stats,
        top_co_occurrences = co_occurrences,
        top_mutual_info = mutual_info_pairs,
        text_stats = TextStats(avg_chars = avg_chars, median_chars = med_chars, p95_chars = p95_chars),
        imbalance = ImbalanceMetrics(gini_coefficient = gini, head_genre_ratio = head_ratio, \
            tail_genre_count = tail_count),
        anomalies = AnomalyReport(duplicate_movies = dup_count, \
            extremely_short_overviews = short_count, movies_with_too_many_genres = too_many_genres, \
                near_duplicate_pairs = len(duplicates)),
        warnings = warnings
    )    
    if output_path:
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with open(output_path, "w", encoding = "utf-8") as f:
            json.dump(report.model_dump(mode = "json"), f, indent = 2)
        print(f"✅ EDA report saved to {output_path}")
        # Upload to cloud storage if configured
        if storage_manager:
            print(f"☁️  Uploading EDA report to cloud storage...")
            storage_manager.save_artifact(output_path, "eda_report.json")
    
    if create_plots and output_path:
        plots_dir: Path = output_path.parent / "eda_plots"
        create_visualizations(movies_df, genres_df, genre_stats, plots_dir)
        if storage_manager:
            print(f"☁️  Uploading EDA plots to cloud storage...")
            for plot_file in plots_dir.glob("*.png"):
                rel_path: str = f"eda_plots/{plot_file.name}"
                storage_manager.save_artifact(plot_file, rel_path)
            print(f"✅ EDA artifacts uploaded successfully")
    # Print summary
    print(f"\n📈 Summary: {n_movies} movies, {n_genres} genres")
    print(f"  Label Density: {density:.3f}, Cardinality: {cardinality}")
    for w in warnings: print(f"  ⚠️  {w}")
    return report

def main() -> None:
    data_dir = Path("IMDb-dataset")
    output_dir = Path("artifacts")
    storage_manager: Optional[StorageManager] = None
    try:
        storage_manager = StorageManager.create_from_env()
    except Exception as e:
        print(f"⚠️  Cloud storage initialization failed: {e}")
        print("  Continuing with local storage only...")
    movies_df, genres_df = load_and_clean_data(data_dir / "movies_overview.csv", data_dir / "movies_genres.csv")
    print(f"✅ Cleaned dataset: {len(movies_df)} movies retained.")
    eda_report: EDAResults = run_eda(movies_df, genres_df, output_path = \
        output_dir / "eda_report.json", create_plots = True, storage_manager = storage_manager)

if __name__ == "__main__": main()