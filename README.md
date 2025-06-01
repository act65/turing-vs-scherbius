# Scherbius versus Turing

__Arthur Scherbius__ built the Engima encryption machine. It was used in World War 2 to encrypt German military communications. __Alan Turing__ is credited with 'breaking' the Enigma encrpytion and using the intel to win the war. 

> Alan Turing: We need your help, to keep this a secret from Admiralty, Army, RAF, uh...as no one can know, that we've broken Enigma, not even Dennison.<br>
...<br>
Alan Turing: While we develop a system to help you determine how much intelligence to act on. Which, uh, attacks to stop, which to let through. Statistical analysis, the minimum number of actions it will take, for us to win the war - but the maximum number we can take before the Germans get suspicious.
(quote from The Imitation Game)

To win this game as 'Turing', this game requires you to;
- A) break a code,
- B) exploit the broken code, without revealing you have broken it. 

# Overview

The game is a turn-based, two player game.
One player plays as 'Turing' and the other as 'Scherbius'.

It has 1 number of hidden information.
The game is played with cards.

## Game logic

The idea for this game is from a friend (Nick Johnstone aka Widdershin).
https://replit.com/@Widdershin/TuringVsScherbius#main.rb

I have implemented the core game in rust with some small changes;

- encryption is based on a 'simple' version of enimga. Aka, a 2 rotor polynumeric substitution cipher.
- re-encryption now costs victory points
- you can send as many or as few resources to a single battle as you like (rather than max 2).

See 'src/' for the rust code and game logic.

## RL agent

I have implemented a RL agent to play the game. 


This is in the 'rl/' directory.


## Pygame interface

I have also implemented a pygame interface for the game which supports the following features;

- using custom images for the game pieces.
- 

See 'py/' for the game interface.

## Deployment to Google Cloud Platform

This project can be deployed to Google Cloud Platform using Cloud Run and Artifact Registry, orchestrated with Terraform.

### Prerequisites

*   **Google Cloud SDK (`gcloud`)**: Install from [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install) and authenticate with your GCP account (`gcloud auth login`, `gcloud auth application-default login`).
*   **Docker**: Install from [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/).
*   **Terraform**: Install from [https://learn.hashicorp.com/tutorials/terraform/install-cli](https://learn.hashicorp.com/tutorials/terraform/install-cli).
*   **Enable GCP APIs**: Ensure the following APIs are enabled in your GCP project:
    *   Artifact Registry API (`artifactregistry.googleapis.com`)
    *   Cloud Run API (`run.googleapis.com`)
    *   Cloud Build API (`cloudbuild.googleapis.com`) (Cloud Build is often used by Cloud Run for deployments, even if not explicitly building via Cloud Build)
    You can enable them via the GCP console or using `gcloud services enable [API_NAME]`.

### Deployment Steps

1.  **Configure GCP Project and Region:**
    Open the `deploy.sh` script and set your Google Cloud Project ID and Region at the top of the file:
    ```bash
    GCP_PROJECT_ID="your-gcp-project-id"
    GCP_REGION="your-gcp-region" # e.g., us-central1
    ```
    You may also customize `ARTIFACT_REGISTRY_REPO_ID` and `CLOUD_RUN_SERVICE_NAME` if desired, though the defaults should work fine.

2.  **Run the Deployment Script:**
    Navigate to the root of the repository in your terminal and execute the script:
    ```bash
    ./deploy.sh
    ```
    The script will:
    *   Build the Docker image for the Flask application.
    *   Push the image to Google Artifact Registry.
    *   Initialize Terraform.
    *   Apply the Terraform configuration to create the Artifact Registry repository (if it doesn't exist) and deploy the application to Cloud Run.

3.  **Access the Application:**
    Once the script completes, it will output the URL of your deployed Cloud Run service. You can access the application by navigating to this URL in your web browser.

### Cleaning Up

To remove the deployed resources:

1.  Navigate to the `terraform` directory:
    ```bash
    cd terraform
    ```
2.  Run `terraform destroy`, providing your project ID and region:
    ```bash
    terraform destroy -var="gcp_project_id=your-gcp-project-id" -var="gcp_region=your-gcp-region"
    ```
3.  You may also want to manually delete the Docker image from Artifact Registry if you no longer need it.

## Code Overview (tvs_flask)

The `tvs_flask` application provides a web-based interface for playing the Turing vs. Scherbius game. It allows users to select a role (Turing or Scherbius) and play against an AI opponent.

### Key Components:

*   **`app.py`**: This is the main Flask application file. It defines the web routes, handles HTTP requests from the client, and uses the `GameManager` to process game logic and serve the HTML templates for the user interface.
*   **`manager.py`**: This module contains the `GameManager` class. The `GameManager` is responsible for orchestrating the game flow, managing the game state (including interactions with the `tvs_core` game engine), handling player actions, and interfacing with the AI logic.
*   **`utils.py`**: This utility module provides helper functions used across the application, primarily by `manager.py`. This includes the AI logic for both Turing and Scherbius, functions for preparing data for client display, and managing game state components.
*   **`static/`**: This directory contains static files served directly to the client, such as `style.css` for styling and `script.js` for client-side interactions.
*   **`templates/`**: This directory holds the HTML templates (e.g., `index.html`) that are rendered by Flask to create the web pages users interact with.
*   **`tests/`**: This directory contains the unit tests for the `tvs_flask` application, written for the pytest framework.

## Python Application Setup (tvs_flask)

These instructions are for setting up and running the Python Flask web application (`tvs_flask`).

1.  **Create and activate a virtual environment:**
    It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    Install Flask and pytest using pip:
    ```bash
    pip install Flask pytest
    ```

## Running Tests (tvs_flask)

To run the tests for the Flask application, navigate to the root of the repository and use the following command:

```bash
python -m pytest tvs_flask/tests
```

**Note on `tvs_core`:** The tests for `tvs_flask` interact with a module named `tvs_core`, which contains the core game logic. This module is expected to be provided or built separately (e.g., from Rust code). If `tvs_core` is not available in your Python environment, you may encounter errors during test collection or execution (specifically `AttributeError: module 'tvs_core' has no attribute 'PyGameState'`). These particular errors can be disregarded if you are only focusing on the `tvs_flask` application components, as the core game logic is mocked in many of the tests.

# Installation

First, clone the repository;

```bash
git clone ...
cd turing-vs-scherbius
```

The source for the game is written in rust. To build the game, you will need to have rust and pyo3 installed.

To install the game, run the following command;

```bash
pip install pyo3
maturin init
maturin develop
```