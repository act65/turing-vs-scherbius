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
PLACEHOLDER_IMAGE="gcr.io/cloudrun/hello" # Define a placeholder image

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
echo "Image Name for app: $IMAGE_NAME:$IMAGE_TAG" # Clarified this is the app image
echo "Placeholder Image for initial Cloud Run: $PLACEHOLDER_IMAGE"
echo "---------------------------"

# --- Initialize Terraform and Create Repository (and other non-Cloud Run infra) ---
echo "--- Initializing Terraform ---"
cd terraform
terraform init -upgrade
echo "--- Applying Terraform to create Artifact Registry Repository AND initial Cloud Run service with placeholder ---"
terraform apply -auto-approve \
  -var="gcp_project_id=${GCP_PROJECT_ID}" \
  -var="gcp_region=${GCP_REGION}" \
  -var="artifact_registry_repository_id=${ARTIFACT_REGISTRY_REPO_ID}" \
  -var="cloud_run_service_name=${CLOUD_RUN_SERVICE_NAME}" \
  -var="cloud_run_image_uri=${PLACEHOLDER_IMAGE}" # Pass placeholder image

cd .. # Back to root

# --- Build Docker Image ---
echo "--- Building Docker Image ---"
FULL_IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_ID}/${CLOUD_RUN_SERVICE_NAME}:${IMAGE_TAG}"
docker build -t "${FULL_IMAGE_NAME}" . # Assuming Dockerfile is in root

# --- Configure Docker for Artifact Registry ---
echo "--- Configuring Docker for Artifact Registry ---"
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# --- Pushing Docker Image to Artifact Registry ---
echo "--- Pushing Docker Image to Artifact Registry ---"
docker push "${FULL_IMAGE_NAME}"

# --- Deploy to Cloud Run using Terraform (now that image exists) ---
echo "--- Applying Terraform to update Cloud Run Service with the actual application image ---"
cd terraform
terraform apply -auto-approve \
  -var="gcp_project_id=${GCP_PROJECT_ID}" \
  -var="gcp_region=${GCP_REGION}" \
  -var="artifact_registry_repository_id=${ARTIFACT_REGISTRY_REPO_ID}" \
  -var="cloud_run_service_name=${CLOUD_RUN_SERVICE_NAME}" \
  -var="cloud_run_image_uri=${FULL_IMAGE_NAME}" # Pass the full image URL to Terraform
cd ..

SERVICE_URL=$(terraform -chdir=terraform output -raw cloud_run_service_url 2>/dev/null || echo "Could not retrieve service URL automatically.")
echo "Deployment complete."
if [ -n "$SERVICE_URL" ] && [ "$SERVICE_URL" != "Could not retrieve service URL automatically." ]; then
    echo "Access at: $SERVICE_URL"
else
    echo "Please check the GCP console for the Cloud Run service URL."
    echo "Ensure your Terraform output 'cloud_run_service_url' is correctly configured."
fi