"""
Main FastAPI application entry point.
"""

# References: https://github.com/tuangatech/MLOps-Projects/blob/master/MLFlow-FastAPI-Fargate/app/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.utils import (
    convert_input_to_lightgbm_data,
    load_ranker_model,
    load_schema,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the model directory relative to this file
MODEL_DIR = "app/model"
logger.info(f"Using model directory: {MODEL_DIR}")

# Load the model and input/output schemas
PredictionRequest, ModelOutput = load_schema(MODEL_DIR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    logger.info("Starting application lifespan...")
    try:
        # Load the ML model
        logger.info("Loading model...")
        app.state.model = load_ranker_model(MODEL_DIR)
        logger.info("Model loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Error during lifespan: {str(e)}")
        raise

    yield
    # Clean up the ML models and release the resources
    logger.info("Cleaning up resources...")
    if hasattr(app.state, "model"):
        del app.state.model


# Define application
app = FastAPI(
    title="Personalized Fashion Recommendation",
    description="Predict personalized fashion recommendations using a LightGBM lambda ranker model",
    version="1.0",
    lifespan=lifespan,
)


# Prediction endpoint
@app.post("/predict", response_model=ModelOutput)
def predict(request: PredictionRequest):
    try:
        logger.info("Received request")
        # Ensure the model is loaded
        if not hasattr(app.state, "model") or app.state.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Convert input to model format
        input_data = request.dict()
        lgb_data = convert_input_to_lightgbm_data(input_data)

        # Make prediction
        prediction = app.state.model.predict_scores(lgb_data)

        # Prepare response with metadata
        return ModelOutput(
            prediction=float(prediction[0]),
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
def health_check():
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    logger.info(f"Health check - Model loaded: {model_loaded}")
    return {"status": "healthy", "model_loaded": model_loaded}


@app.get("/ready")
def readiness_check():
    logger.info("Readiness check")
    # Perform deep health checks (e.g., DB, ML model)
    if not hasattr(app.state, "model") or app.state.model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Dependencies not ready")
    return {"status": "ready"}
