#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# !!! IMPORTANT !!!
# SET YOUR GCP PROJECT ID AND REGION BELOW
GCP_PROJECT_ID="turing-vs-scherbius"
GCP_REGION="australia-southeast2"

# Default values (can be overridden by environment variables if needed)
ARTIFACT_REGISTRY_REPO_ID="${ARTIFACT_REGISTRY_REPO_ID:-tvs-app-repo}"
CLOUD_RUN_SERVICE_NAME="${CLOUD_RUN_SERVICE_NAME:-tvs-flask-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Construct the full image name
IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_ID}/${CLOUD_RUN_SERVICE_NAME}"

echo "--- Starting Deployment ---"
echo "GCP Project ID: $GCP_PROJECT_ID"
echo "GCP Region: $GCP_REGION"
echo "Artifact Registry Repo ID: $ARTIFACT_REGISTRY_REPO_ID"
echo "Cloud Run Service Name: $CLOUD_RUN_SERVICE_NAME"
echo "Image Name for app: $IMAGE_NAME:$IMAGE_TAG" # Clarified this is the app image
echo "---------------------------"

# --- Build Docker Image ---
echo "--- Building Docker Image ---"
FULL_IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_ID}/${CLOUD_RUN_SERVICE_NAME}:${IMAGE_TAG}"
docker build -t "${FULL_IMAGE_NAME}" . # Assuming Dockerfile is in root

# --- Running Docker Image ---
docker run -it -p 5000:5000 "${FULL_IMAGE_NAME}"