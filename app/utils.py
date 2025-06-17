from typing import Any, Dict, Optional, Tuple, Type

import mlflow
import pandas as pd
from lightgbm import (  # noqa: F401 (to avoid unused import warning, this is necessary to segfault)
    LGBMRanker,
)
from mlflow.models import Model
from pydantic import BaseModel, Field, RootModel, create_model

from src.input_preprocessing import LightGBMDataResult

# Mapping MLflow data types to Python types
_MLFLOW_TYPE_MAP = {
    "double": float,
    "float": float,
    "integer": int,
    "long": int,
    "boolean": bool,
    "string": str,
}


# ======================================================================================================================
# Model input and output
# ======================================================================================================================
class ModelInput(RootModel):
    """Base model for the actual model input features."""

    root: Any  # This will be set dynamically based on the model's input schema


class PredictionRequest(BaseModel):
    """Request model for predictions with metadata."""

    data: ModelInput = Field(..., description="Model input features")
    feature_names: Dict[str, list] = Field(..., description="Mapping of feature types to feature names")
    sample: Optional[str] = Field(None, description="Sample identifier or description")


class ModelOutput(BaseModel):
    """Base model for model predictions."""

    prediction: float


def load_ranker_model(model_path: str) -> object:
    """
    Loads an MLflow model.
    """
    return mlflow.lightgbm.load_model(model_path)


def load_schema(model_path: str) -> Tuple[Type[BaseModel], Type[BaseModel]]:
    """
    Loads an MLflow model and dynamically generates Pydantic models
    from the input and output signatures defined in the MLmodel file.

    Args:
        model_path (str): Path to the MLflow model directory.

    Returns:
        Tuple containing:
            - A dynamically created Pydantic model class for input validation
            - A Pydantic model class for output validation
    """
    # Load MLflow model
    model_metadata = Model.load(model_path)

    # Get model signature
    signature = model_metadata.signature
    if signature is None:
        raise ValueError("MLmodel has no signature block.")

    # Handle input schema
    input_schema = signature.inputs.to_dict()
    if not input_schema:
        raise ValueError("Model signature does not define any inputs.")

    # Build fields for input Pydantic model
    input_fields = {}
    for col in input_schema:
        col_name = col["name"]
        mlflow_type = col["type"]
        py_type = _MLFLOW_TYPE_MAP.get(mlflow_type)
        if py_type is None:
            raise TypeError(f"Unsupported MLflow type '{mlflow_type}' for column '{col_name}'")
        input_fields[col_name] = (py_type, ...)  # "..." = required

    # Dynamically create input Pydantic model
    BaseInputModel = create_model("DynamicInput", **input_fields)

    # Set the root type for ModelInput
    ModelInput.model_rebuild()
    ModelInput.__annotations__["root"] = BaseInputModel

    return PredictionRequest, ModelOutput


# ======================================================================================================================
# Helper functions for input processing
# ======================================================================================================================
def convert_input_to_lightgbm_data(input: dict) -> LightGBMDataResult:
    """
    Converts input data to LightGBM data format.
    """
    # Convert input to pandas DataFrame
    input_df = pd.DataFrame([input["data"]])
    return LightGBMDataResult(
        data=input_df, use_type="inference", feature_names=input["feature_names"], sample=input["sample"]
    )
