#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (one level up from scripts)
# Note that we change the working directory to the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Source the .env file from project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
else
    echo "Error: .env file not found in $PROJECT_ROOT"
    exit 1
fi

ECR_REPO_URL="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY_NAME"
echo "ECR Repo URL: $ECR_REPO_URL"

echo "Building image"
docker build --no-cache --platform linux/amd64 -t $IMAGE_NAME -f "$PROJECT_ROOT/Dockerfile" "$PROJECT_ROOT"

echo "Logging in to ECR"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Tagging image"
docker tag $IMAGE_NAME:$IMAGE_TAG $ECR_REPO_URL:$IMAGE_TAG

echo "Pushing image to ECR"
docker push $ECR_REPO_URL:$IMAGE_TAG

echo "Successfully deployed image to $ECR_REPO_URL:$IMAGE_TAG"
