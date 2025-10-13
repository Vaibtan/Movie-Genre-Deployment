import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from bin_relevance_lr import train_binary_relevance
from data_loader import load_and_clean_data
from multi_label_knn import train_ml_knn


def train_all_models(data_dir: Path, artifact_dir: Path, train_br: bool = True, \
    train_mlknn: bool = True, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Args:
        data_dir: Directory containing dataset
        artifact_dir: Directory to save artifacts
        train_br: Whether to train Binary Relevance
        train_mlknn: Whether to train ML-kNN
        test_size: Validation split proportion
        random_state: Random seed
    Returns: Dictionary mapping model names to their metrics        
    """
    results: Dict[str, Dict[str, float]] = {}
    
    print("\n" + "="*70)
    print("Multi-Label Genre Classification - Training Pipeline")
    print("="*70)
    print(f"\n📂 Loading data from {data_dir}...")
    try:
        movies_df, genres_df = load_and_clean_data(data_dir / "movies_overview.csv", \
            data_dir / "movies_genres.csv")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nPlease ensure the dataset is in the correct location:")
        print(f"  {data_dir / 'movies_overview.csv'}")
        print(f"  {data_dir / 'movies_genres.csv'}")
        sys.exit(1)
    
    # Remove duplicates
    initial_count: int = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset = ["title", "overview"]).reset_index(drop = True)
    removed_count: int = initial_count - len(movies_df)
    if removed_count > 0: print(f"  Removed {removed_count} duplicate entries")
    print(f"\n✅ Dataset ready: {len(movies_df)} movies, {len(genres_df)} genres")
    
    # Train Binary Relevance
    if train_br:
        print("\n" + "="*70)
        print("1️⃣  Training Binary Relevance (Problem Transformation)")
        print("="*70)
        
        try:
            start_time: float = time.time() 
            br_metrics = train_binary_relevance(
                movies_df = movies_df,
                genres_df = genres_df,
                test_size = test_size,
                random_state = random_state,
                artifact_dir = artifact_dir,
                max_features = 20000,
                C = 1.0,
            )
            
            elapsed: float = time.time() - start_time
            print(f"\n⏱️  Binary Relevance training completed in {elapsed:.1f}s")
            results["binary_relevance"] = br_metrics
            
        except Exception as e:
            print(f"\n❌ Binary Relevance training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train ML-kNN
    if train_mlknn:
        print("\n" + "="*70)
        print("2️⃣  Training ML-kNN (Algorithmic Adaptation)")
        print("="*70)
        try:
            start_time = time.time()
            mlknn_metrics = train_ml_knn(movies_df = movies_df, genres_df = genres_df, \
                test_size = test_size, random_state = random_state, artifact_dir = artifact_dir, \
                    max_features = 20000, k = 10, s = 1.0)
            elapsed = time.time() - start_time
            print(f"\n⏱️  ML-kNN training completed in {elapsed:.1f}s")
            results["ml_knn"] = mlknn_metrics
        except Exception as e:
            print(f"\n ML-kNN training failed: {e}")
            import traceback
            traceback.print_exc()
    return results

def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    if not results: print("\n⚠️  No models were trained successfully"); return
    print("\n" + "="*70)
    print("📊 Model Comparison")
    print("="*70)    
    all_metrics = set()
    for metrics in results.values(): all_metrics.update(metrics.keys())
    metric_order = ["subset_accuracy", "hamming_loss", "f1_micro", "f1_macro", "precision_at_3", "recall_at_3"]
    print(f"\n{'Metric':<20} ", end="")
    for model_name in results.keys():
        display_name = model_name.replace("_", " ").title()
        print(f"{display_name:>15} ", end="")
    print()
    print("-" * 70)
    for metric in metric_order:
        if metric not in all_metrics: continue
        display_metric = metric.replace("_", " ").title()
        print(f"{display_metric:<20} ", end="")
        values = []
        for model_name in results.keys():
            if metric in results[model_name]:
                value = results[model_name][metric]
                values.append(value)
                print(f"{value:>15.4f} ", end="")
            else: print(f"{'N/A':>15} ", end="")        
        if values and metric != "hamming_loss":
            best_idx = values.index(max(values))
            print(f"  ← Best", end="")
        elif values and metric == "hamming_loss":
            best_idx = values.index(min(values))
            print(f"  ← Best", end="")
        print()

def main() -> None:
    parser = argparse.ArgumentParser(description = "Train multi-label genre classification models")    
    parser.add_argument(
        "--data-dir",
        type = Path,
        default = Path("IMDb-dataset"),
        help = "Directory containing dataset (default: IMDb-dataset)",
    )
    
    parser.add_argument(
        "--artifact-dir",
        type = Path,
        default = Path("artifacts"),
        help = "Directory to save model artifacts (default: artifacts)",
    )
    
    parser.add_argument(
        "--model",
        choices = ["br", "mlknn", "all"],
        default = "all",
        help = "Which model to train: br (Binary Relevance), mlknn, or all (default: all)",
    )
    
    parser.add_argument(
        "--test-size",
        type = float,
        default = 0.2,
        help = "Validation split proportion (default: 0.2)",
    )
    
    parser.add_argument(
        "--random-state",
        type = int,
        default = 42,
        help = "Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    train_br: bool = args.model in ["br", "all"]
    train_mlknn: bool = args.model in ["mlknn", "all"]    
    overall_start: float = time.time()
    results = train_all_models(data_dir = args.data_dir, artifact_dir =args.artifact_dir, \
        train_br = train_br, train_mlknn = train_mlknn, test_size = args.test_size, random_state = args.random_state,)
    overall_elapsed: float = time.time() - overall_start    
    print_comparison(results)
    
    # Print summary
    print("\n" + "="*70)
    print("🎉 Training Pipeline Complete!")
    print("="*70)
    print(f"\n⏱️  Total time: {overall_elapsed:.1f}s")
    print(f"📁 Artifacts saved to: {args.artifact_dir}")
    print(f"🚀 Ready to serve models via API!")

if __name__ == "__main__": main()