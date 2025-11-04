# Dockerfile for the GeoAI Streamlit Application

# --- Base Stage ---
# Use an NVIDIA CUDA base image that includes Python and development tools
# This supports the GPU requirements of the ML models (torch, bitsandbytes)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - python3-pip for package management
# - libgdal-dev and gdal-bin for geospatial libraries (geopandas, rasterio)
# - git for any potential VCS dependencies
RUN apt-get update \ 
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    libgdal-dev \
    gdal-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- Builder Stage ---
# This stage installs Python dependencies
FROM base AS builder

WORKDIR /app

# First, install PyTorch with CUDA support. This is crucial for GPU acceleration.
# Installing it separately ensures the correct version is pulled from NVIDIA's index.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create and copy requirements.txt to leverage Docker layer caching
COPY requirements.txt .

# Install the rest of the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# --- Final Stage ---
# This is the final, production-ready image
FROM base AS final

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the path to the venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

# Create the logs directory and set appropriate permissions
RUN mkdir logs \
    && chown -R 1000:1000 /app

# Switch to a non-root user for better security
USER 1000

# Expose the default Streamlit port
EXPOSE 8501

# The command to run the Streamlit application
# --server.address=0.0.0.0 is necessary to access the app from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]