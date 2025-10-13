import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from model_loader import ModelLoader
from monitoring import (INFERENCE_TIME, PREDICTION_CONFIDENCE, REQUEST_VOLUME,
                        PrometheusMiddleware)
from src.schema import PredictRequest, PredictResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger: logging.Logger = logging.getLogger("genre-api")

app = FastAPI(
    title="Movie Genre Prediction API",
    version="1.0.0",
    description="Multi-label genre prediction using transformation and adaptation methods",
)
app.add_middleware(PrometheusMiddleware)

ARTIFACTS_DIR: Path = Path("artifacts")

# Lazy-load models (initialized on first request)
_br_loader: ModelLoader | None = None
_mlknn_loader: ModelLoader | None = None


def get_br_loader() -> ModelLoader:
    """Lazy initialization of Binary Relevance model."""
    global _br_loader
    if _br_loader is None:
        logger.info("Loading Binary Relevance model...")
        _br_loader = ModelLoader(
            model_path=ARTIFACTS_DIR / "binary_relevance_model.pkl",
            vectorizer_path=ARTIFACTS_DIR / "tfidf_vectorizer.pkl",
            genre_names_path=ARTIFACTS_DIR / "genre_names.txt",
        )
        logger.info("Binary Relevance model loaded successfully")
    return _br_loader


def get_mlknn_loader() -> ModelLoader:
    """Lazy initialization of ML-kNN model."""
    global _mlknn_loader
    if _mlknn_loader is None:
        logger.info("Loading ML-kNN model...")
        _mlknn_loader = ModelLoader(
            model_path=ARTIFACTS_DIR / "mlknn_model.pkl",
            vectorizer_path=ARTIFACTS_DIR / "tfidf_vectorizer_mlkn.pkl",
            genre_names_path=ARTIFACTS_DIR / "genre_names_mlkn.txt",
        )
        logger.info("ML-kNN model loaded successfully")
    return _mlknn_loader


@app.get("/", status_code=status.HTTP_200_OK)
async def root() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "Movie Genre Prediction API",
            "version": "1.0.0",
        }
    )


@app.post("/predict/transformation", response_model=PredictResponse)
async def predict_transformation(request: PredictRequest) -> PredictResponse:
    """
    Predict genres using Binary Relevance (Problem Transformation).
    
    Args:
        request: PredictRequest with movie overview
        
    Returns:
        PredictResponse with predicted genres and inference time
        
    Raises:
        HTTPException: If prediction fails
    """
    model_type: str = "transformation"
    
    try:
        REQUEST_VOLUME.labels(model_type=model_type).inc()
        
        # Load model lazily
        br_loader: ModelLoader = get_br_loader()
        
        # Perform prediction
        genres: List[str]
        latency: float
        confidences: npt.NDArray[np.float64]
        genres, latency, confidences = br_loader.predict(request.overview, k=3)
        
        # Log inference time
        INFERENCE_TIME.labels(model_type=model_type).observe(latency)
        
        # Log top-3 confidences (FIXED: correct indexing)
        top_indices: npt.NDArray[np.intp] = np.argsort(-confidences)[:3]
        for idx in top_indices:
            if idx < len(br_loader.genre_names) and confidences[idx] > 0.1:
                genre_name: str = br_loader.genre_names[idx]
                confidence: float = float(confidences[idx])
                PREDICTION_CONFIDENCE.labels(
                    model_type=model_type, genre=genre_name
                ).observe(confidence)
        
        logger.info(
            f'{{"model": "{model_type}", "genres": {genres}, '
            f'"latency_ms": {latency * 1000:.2f}}}'
        )
        
        return PredictResponse(
            predicted_genres=genres,
            inference_time_sec=latency,
            model_type=model_type,
            confidence_scores=[float(confidences[i]) for i in top_indices],
        )
        
    except FileNotFoundError as e:
        logger.error(f'{{"error": "Model files not found", "exception": "{str(e)}"}}')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please ensure models are trained.",
        ) from e
        
    except Exception as e:
        logger.error(
            f'{{"error": "Prediction failed", "exception": "{str(e)}", '
            f'"model": "{model_type}"}}'
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.post("/predict/adaptation", response_model=PredictResponse)
async def predict_adaptation(request: PredictRequest) -> PredictResponse:
    """
    Predict genres using ML-kNN (Algorithmic Adaptation).
    
    Args:
        request: PredictRequest with movie overview
        
    Returns:
        PredictResponse with predicted genres and inference time
        
    Raises:
        HTTPException: If prediction fails
    """
    model_type: str = "adaptation"
    
    try:
        REQUEST_VOLUME.labels(model_type=model_type).inc()
        
        # Load model lazily
        mlknn_loader: ModelLoader = get_mlknn_loader()
        
        # Perform prediction
        genres: List[str]
        latency: float
        confidences: npt.NDArray[np.float64]
        genres, latency, confidences = mlknn_loader.predict(request.overview, k=3)
        
        # Log inference time
        INFERENCE_TIME.labels(model_type=model_type).observe(latency)
        
        # Log top-3 confidences
        top_indices: npt.NDArray[np.intp] = np.argsort(-confidences)[:3]
        for idx in top_indices:
            if idx < len(mlknn_loader.genre_names) and confidences[idx] > 0.1:
                genre_name: str = mlknn_loader.genre_names[idx]
                confidence: float = float(confidences[idx])
                PREDICTION_CONFIDENCE.labels(
                    model_type=model_type, genre=genre_name
                ).observe(confidence)
        
        logger.info(
            f'{{"model": "{model_type}", "genres": {genres}, '
            f'"latency_ms": {latency * 1000:.2f}}}'
        )
        
        return PredictResponse(
            predicted_genres=genres,
            inference_time_sec=latency,
            model_type=model_type,
            confidence_scores=[float(confidences[i]) for i in top_indices],
        )
        
    except FileNotFoundError as e:
        logger.error(f'{{"error": "Model files not found", "exception": "{str(e)}"}}')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please ensure models are trained.",
        ) from e
        
    except Exception as e:
        logger.error(
            f'{{"error": "Prediction failed", "exception": "{str(e)}", '
            f'"model": "{model_type}"}}'
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from e


@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.
    
    Returns:
        Response with Prometheus-formatted metrics
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
async def health() -> JSONResponse:
    """
    Detailed health check with model status.
    
    Returns:
        JSONResponse with service health status
    """
    br_loaded: bool = _br_loader is not None
    mlknn_loaded: bool = _mlknn_loader is not None
    
    return JSONResponse(
        content={
            "status": "healthy" if (br_loaded or mlknn_loaded) else "degraded",
            "models": {
                "binary_relevance": "loaded" if br_loaded else "not_loaded",
                "ml_knn": "loaded" if mlknn_loaded else "not_loaded",
            },
        }
    )