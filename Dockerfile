# Use python:3.9-slim as the base image
FROM python:3.9-slim

# Set environment variables to ensure Rust is installed in a non-interactive way
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# Install Rust and Cargo
RUN apt-get update && apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the WORKDIR to /app
WORKDIR /app

# Install build dependencies for maturin and the Rust project
RUN pip install maturin wheel pytest

# Install Flask
RUN pip install Flask

# Copy the tvs_core directory into the image
COPY tvs_core /app/tvs_core

# Install the tvs_core Rust project as a Python package
RUN cd /app/tvs_core && make

# Copy the tvs_flask directory into the image
COPY tvs_flask /app/tvs_flask

# Change WORKDIR to /app/tvs_flask
WORKDIR /app/tvs_flask

# Expose port 5000 (default for Flask)
EXPOSE 5000

# Set the command to run the Flask app
CMD ["python", "app.py"]
