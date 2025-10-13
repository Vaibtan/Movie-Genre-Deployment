import time

from fastapi import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from starlette.middleware.base import BaseHTTPMiddleware

# Metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["endpoint"])
INFERENCE_TIME = Histogram("model_inference_duration_seconds", "Model inference time", ["model_type"])
PREDICTION_CONFIDENCE = Histogram("model_prediction_confidence", "Prediction confidence scores", ["model_type", "genre"])
REQUEST_VOLUME = Counter("model_requests_total", "Total model prediction requests", ["model_type"])

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        latency = time.time() - start
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
        if request.url.path in ["/predict/transformation", "/predict/adaptation"]:
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
        return response