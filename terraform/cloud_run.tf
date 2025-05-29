resource "google_cloud_run_v2_service" "default" {
  provider = google-beta # Cloud Run v2 might require google-beta
  name     = var.cloud_run_service_name
  location = var.gcp_region
  project  = var.gcp_project_id

  template {
    containers {
      image = var.cloud_run_image_uri
      ports {
        container_port = 5000
      }
      resources {
        limits = {
          memory = "512Mi"
          cpu    = "1"
        }
      }
    }
  }
}

resource "google_cloud_run_v2_service_iam_binding" "allow_unauthenticated" {
  provider = google-beta # IAM for Cloud Run v2 might require google-beta
  project  = google_cloud_run_v2_service.default.project
  location = google_cloud_run_v2_service.default.location
  name     = google_cloud_run_v2_service.default.name
  role     = "roles/run.invoker"
  members  = ["allUsers"]
}

output "cloud_run_service_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.default.uri
}
