variable "gcp_project_id" {
  description = "The GCP project ID to deploy resources to."
  type        = string
  default     = "YOUR_GCP_PROJECT_ID" # User should replace this
}

variable "gcp_region" {
  description = "The GCP region to deploy resources to."
  type        = string
  default     = "YOUR_GCP_REGION" # User should replace this
}

variable "artifact_registry_repository_id" {
  description = "The ID for the Artifact Registry repository."
  type        = string
  default     = "tvs-app-repo"
}

variable "cloud_run_service_name" {
  description = "The name for the Cloud Run service."
  type        = string
  default     = "tvs-flask-app"
}

variable "cloud_run_image_uri" {
  description = "The full URI of the Docker image to deploy to Cloud Run."
  type        = string
  # No default, as this will be constructed and passed in by the deploy script.
}