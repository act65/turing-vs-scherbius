#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# !!! IMPORTANT !!!
# SET YOUR GCP PROJECT ID AND REGION BELOW
GCP_PROJECT_ID="YOUR_GCP_PROJECT_ID"
GCP_REGION="YOUR_GCP_REGION"

# Default values (can be overridden by environment variables if needed)
ARTIFACT_REGISTRY_REPO_ID="${ARTIFACT_REGISTRY_REPO_ID:-tvs-app-repo}"
CLOUD_RUN_SERVICE_NAME="${CLOUD_RUN_SERVICE_NAME:-tvs-flask-app}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Construct the full image name
IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_ID}/${CLOUD_RUN_SERVICE_NAME}"

# --- Prerequisite Checks ---
echo "Checking prerequisites..."

if ! command -v gcloud &> /dev/null
then
    echo "Error: gcloud CLI not found. Please install and configure it."
    exit 1
fi

if ! command -v docker &> /dev/null
then
    echo "Error: Docker CLI not found. Please install it."
    exit 1
fi

if ! command -v terraform &> /dev/null
then
    echo "Error: Terraform CLI not found. Please install it."
    exit 1
fi
echo "Prerequisites met."

# --- User Confirmation for GCP Project and Region ---
if [ "$GCP_PROJECT_ID" == "YOUR_GCP_PROJECT_ID" ] || [ "$GCP_REGION" == "YOUR_GCP_REGION" ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! ERROR: You must set GCP_PROJECT_ID and GCP_REGION in this script. !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
fi

echo "--- Starting Deployment ---"
echo "GCP Project ID: $GCP_PROJECT_ID"
echo "GCP Region: $GCP_REGION"
echo "Artifact Registry Repo ID: $ARTIFACT_REGISTRY_REPO_ID"
echo "Cloud Run Service Name: $CLOUD_RUN_SERVICE_NAME"
echo "Image Name: $IMAGE_NAME:$IMAGE_TAG"
echo "---------------------------"

# --- Build and Push Docker Image ---
echo ""
echo "--- Building Docker Image ---"
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "--- Authenticating Docker with GCP Artifact Registry ---"
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
echo "Docker authenticated."

echo ""
echo "--- Pushing Docker Image to Artifact Registry ---"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"
echo "Docker image pushed successfully to ${IMAGE_NAME}:${IMAGE_TAG}"

# --- Deploy with Terraform ---
echo ""
echo "--- Deploying Infrastructure with Terraform ---"
cd terraform

echo "Initializing Terraform..."
terraform init -input=false
echo "Terraform initialized."

echo "Applying Terraform configuration..."
terraform apply -auto-approve \
    -var="gcp_project_id=${GCP_PROJECT_ID}" \
    -var="gcp_region=${GCP_REGION}" \
    -var="artifact_registry_repository_id=${ARTIFACT_REGISTRY_REPO_ID}" \
    -var="cloud_run_service_name=${CLOUD_RUN_SERVICE_NAME}" \
    -var="cloud_run_image_uri=${IMAGE_NAME}:${IMAGE_TAG}"
echo "Terraform apply completed."

# --- Output Service URL ---
echo ""
echo "--- Fetching Cloud Run Service URL ---"
SERVICE_URL=$(terraform output -raw cloud_run_service_url)

if [ -z "$SERVICE_URL" ]; then
    echo "Could not retrieve the service URL. Please check the Cloud Run console."
else
    echo "Cloud Run service deployed successfully!"
    echo "Service URL: ${SERVICE_URL}"
fi

cd ..
echo ""
echo "--- Deployment Script Finished ---"
