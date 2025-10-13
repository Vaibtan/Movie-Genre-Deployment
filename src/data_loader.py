import json
from pathlib import Path
from typing import Any, List, Set, Tuple

import pandas as pd


# Safely parse genre_ids from various input formats
def parse_genre_ids(genre_str: Any) -> List[int]:
    # Handle NaN or empty strings
    if pd.isna(genre_str) or genre_str == "": return []
    if isinstance(genre_str, list):
        return [int(x) for x in genre_str if isinstance(x, (int, float, str)) and str(x).isdigit()]
    if isinstance(genre_str, str):
        try:
            cleaned: str = genre_str.strip()            
            if cleaned == "[]": return []
            # JSON list format
            if cleaned.startswith("[") and cleaned.endswith("]"):
                # Replace single quotes with double quotes for JSON
                json_str: str = cleaned.replace("'", '"')
                parsed: List[Any] = json.loads(json_str)
                return [int(x) for x in parsed if isinstance(x, (int, float, str))]
            # Single value
            if cleaned.isdigit(): return [int(cleaned)]
        except (json.JSONDecodeError, ValueError): return []
    return []


def load_and_clean_data(overview_path: Path, genres_path: Path, min_overview_length: int = 5,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not overview_path.exists(): raise FileNotFoundError(f"Overview file not found: {overview_path}")
    if not genres_path.exists(): raise FileNotFoundError(f"Genres file not found: {genres_path}")
    genres_df: pd.DataFrame = pd.read_csv(genres_path)
    movies_df: pd.DataFrame = pd.read_csv(overview_path)
    print(f"Loaded {len(movies_df)} movies and {len(genres_df)} genres")
    # Validate required columns
    required_movie_cols: Set[str] = {"title", "overview", "genre_ids"}
    required_genre_cols: Set[str] = {"id", "name"}
    if not required_movie_cols.issubset(movies_df.columns):
        raise ValueError(
            f"Missing required movie columns. Expected: {required_movie_cols}, "
            f"Found: {set(movies_df.columns)}"
        ) 
    if not required_genre_cols.issubset(genres_df.columns):
        raise ValueError(
            f"Missing required genre columns. Expected: {required_genre_cols}, "
            f"Found: {set(genres_df.columns)}"
        )    
    movies_df["genre_ids_parsed"] = movies_df["genre_ids"].apply(parse_genre_ids)
    # Get valid genre IDs
    valid_ids: Set[int] = set(genres_df["id"].tolist())
    
    def filter_valid_genres(genre_list: List[int]) -> List[int]:
        if not genre_list: return []
        return [gid for gid in genre_list if gid in valid_ids]

    movies_df["cleaned_genres"] = movies_df["genre_ids_parsed"].apply(filter_valid_genres)
    initial_count: int = len(movies_df)    
    mask: pd.Series = (
        movies_df["overview"].notna()
        & (movies_df["overview"].str.len() > min_overview_length)
        & (movies_df["cleaned_genres"].apply(len) > 0)
    )
    movies_df = movies_df[mask].copy()
    # Rename cleaned column
    movies_df = movies_df.drop(columns = ["genre_ids", "genre_ids_parsed"]).rename(columns = {"cleaned_genres": "genre_ids"})
    # Reset index
    movies_df = movies_df.reset_index(drop = True)
    filtered_count: int = len(movies_df)
    print(
        f"Filtered dataset: {filtered_count} movies retained "
        f"({initial_count - filtered_count} removed)"
    )
    return movies_df, genres_df