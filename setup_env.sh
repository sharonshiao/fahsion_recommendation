#!/bin/bash

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="http://localhost:8080"
export MLFLOW_BACKEND_STORE_URI="./model/mlruns"


# Print confirmation message
echo "Environment variables have been set:"
echo "MLFLOW_TRACKING_URI: $MLFLOW_TRACKING_URI"
echo "MLFLOW_BACKEND_STORE_URI: $MLFLOW_BACKEND_STORE_URI"
