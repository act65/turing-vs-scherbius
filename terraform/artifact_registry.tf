resource "google_artifact_registry_repository" "default" {
  provider      = google-beta # Ensure you have the google-beta provider configured
  project       = var.gcp_project_id
  location      = var.gcp_region
  repository_id = var.artifact_registry_repository_id
  description   = "Docker repository for TVS application"
  format        = "DOCKER"
}
