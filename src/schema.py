from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class GenreMapping(BaseModel):
    # Genre ID to name mapping
    id: int = Field(..., description = "Unique genre identifier")
    name: str = Field(..., min_length = 1, description = "Genre name")

class MovieRecord(BaseModel):
    # Single movie record with genres
    title: str = Field(..., min_length = 1, description = "Movie title")
    overview: Optional[str] = Field(None, description = "Movie overview/synopsis")
    genre_ids: List[int] = Field(..., min_items = 1, description = "List of genre IDs")
    
    # Ensure genre IDs are unique and sorted
    @field_validator("genre_ids")
    @classmethod
    def validate_genre_ids(cls, v: List[int]) -> List[int]:
        return sorted(set(v))


class GenreStats(BaseModel):
    # Statistics for a single genre
    genre_name: str = Field(..., description = "Genre name")
    count: int = Field(..., ge = 0, description = "Number of movies with this genre")
    support_ratio: float = Field(..., ge = 0.0, le = 100.0, description = "Percentage of movies with this genre")

class CoOccurrencePair(BaseModel):
    # Co-occurrence statistics for a genre pair
    genre_a: str = Field(..., description = "First genre name")
    genre_b: str = Field(..., description = "Second genre name")
    count: int = Field(..., ge = 0, description = "Co-occurrence count")


class MutualInfoPair(BaseModel):
    # Normalized mutual information between two genres
    genre_a: str = Field(..., description = "First genre name")
    genre_b: str = Field(..., description = "Second genre name")
    nmi_score: float = Field(..., ge = 0.0, le = 1.0, description = "Normalized mutual information score")

class TextStats(BaseModel):
    avg_chars: float = Field(..., ge = 0.0, description = "Average character length")
    median_chars: float = Field(..., ge = 0.0, description = "Median character length")
    p95_chars: float = Field(..., ge = 0.0, description = "95th percentile character length")


class ImbalanceMetrics(BaseModel):
    gini_coefficient: float = Field(..., ge = 0.0, le = 1.0, description = "Gini coefficient of genre distribution")
    head_genre_ratio: float = Field(..., ge = 0.0, le = 1.0, description = "Proportion of labels covered by top 3 genres")
    tail_genre_count: int = Field(..., ge = 0, description = "Number of genres with <1% support")


class LabelDensityMetrics(BaseModel):
    # Multi-label density and cardinality metrics
    label_density: float = Field(..., ge = 0.0, description = "Average number of labels per instance")
    label_cardinality: int = Field(..., ge = 0, description = "Number of unique label combinations")


class AnomalyReport(BaseModel):
    duplicate_movies: int = Field(..., ge = 0, description = "Exact duplicate count")
    extremely_short_overviews: int = Field(..., ge = 0, description = "Movies with <10 char overviews")
    movies_with_too_many_genres: int = Field(..., ge = 0, description = "Movies with >7 genres")
    near_duplicate_pairs: int = Field(default = 0, ge = 0, description = "Near-duplicate pairs (>90% similarity)")


class EDAResults(BaseModel):
    n_movies: int = Field(..., ge = 0, description = "Total number of movies")
    n_genres: int = Field(..., ge = 0, description = "Total number of genres")
    avg_genres_per_movie: float = Field(..., ge = 0.0, description = "Average genres per movie")
    max_genres_in_one_movie: int = Field(..., ge = 0, description = "Maximum genres in a single movie")
    label_density: float = Field(..., ge = 0.0, description = "Label density (avg labels per instance)")
    label_cardinality: int = Field(..., ge = 0, description = "Label cardinality (unique label combinations)")
    genre_distribution: List[GenreStats] = Field(..., description = "Per-genre statistics")
    top_co_occurrences: List[CoOccurrencePair] = Field(..., description = "Most common genre pairs")
    top_mutual_info: List[MutualInfoPair] = Field(default_factory = list, description = "Top genre pairs by mutual information")
    text_stats: TextStats = Field(..., description = "Text length statistics")
    imbalance: ImbalanceMetrics = Field(..., description = "Imbalance metrics")
    anomalies: AnomalyReport = Field(..., description = "Data quality issues")
    warnings: List[str] = Field(default_factory = list, description = "EDA warnings and recommendations")

class PredictRequest(BaseModel):
    # API request for genre prediction
    title: Optional[str] = Field(None, description = "Movie title (optional)")
    overview: str = Field(..., min_length = 10, max_length = 10000, description = "Movie overview/synopsis")
    
    @field_validator("overview")
    @classmethod
    def validate_overview(cls, v: str) -> str:
        if not v.strip(): raise ValueError("Overview cannot be empty or whitespace")
        return v.strip()


class PredictResponse(BaseModel):
    # API response for genre prediction
    predicted_genres: List[str] = Field(..., min_items = 0, max_items = 10, description = "Predicted genre names")
    inference_time_sec: float = Field(..., ge = 0.0, description = "Inference time in seconds")
    model_type: Optional[str] = Field(None, description = "Model type used (transformation/adaptation)")
    confidence_scores: Optional[List[float]] = Field(None, description = "Confidence scores for predicted genres")