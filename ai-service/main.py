"""
E-KYC Platform - FastAPI Backend
Main application entry point with middleware and route registration.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from routes import kyc_router
from models import HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Rate limiting storage (use Redis in production)
_request_counts: dict = {}
RATE_LIMIT = 60  # requests per minute
RATE_WINDOW = 60  # seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting e-KYC AI Service...")
    
    # Pre-load models on startup (optional - speeds up first request)
    # Uncomment these for production:
    # from services import ocr_engine, face_detector, face_verifier
    # ocr_engine._ensure_reader()
    # face_detector._ensure_models()
    # face_verifier._ensure_models()
    
    logger.info("e-KYC AI Service started successfully")
    yield
    logger.info("Shutting down e-KYC AI Service...")


# Create FastAPI app
app = FastAPI(
    title="E-KYC AI Service",
    description="AI-powered electronic Know Your Customer verification platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - configure for your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Clean old entries
    _request_counts[client_ip] = [
        t for t in _request_counts.get(client_ip, [])
        if current_time - t < RATE_WINDOW
    ]
    
    # Check rate limit
    if len(_request_counts.get(client_ip, [])) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded", "detail": "Too many requests"}
        )
    
    # Record request
    if client_ip not in _request_counts:
        _request_counts[client_ip] = []
    _request_counts[client_ip].append(current_time)
    
    # Process request
    response = await call_next(request)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests (sanitized - no PII in logs)."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


# Register routers
app.include_router(kyc_router)


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        services={
            "ocr": "ready",
            "face_detection": "ready",
            "liveness": "ready",
            "face_verification": "ready"
        }
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        services={
            "ocr": "ready",
            "face_detection": "ready",
            "liveness": "ready",
            "face_verification": "ready"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
