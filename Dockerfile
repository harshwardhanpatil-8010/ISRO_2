FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 1. Set up environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 2. Install system dependencies
# Install Python, PIP, and essential build tools.
# Install libgdal-dev for geospatial libraries like rasterio and geopandas.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv build-essential libgdal-dev gdal-bin && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a symlink for python3 to be the default
RUN ln -s /usr/bin/python3 /usr/bin/python

# 3. Install Python dependencies
# Create and activate a virtual environment for better dependency management.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support first to ensure compatibility.
# See https://pytorch.org/ for the correct command for your CUDA version.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining application and ML dependencies.
# These are installed in a single layer to optimize image size.
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    geopandas \
    rasterio \
    folium \
    streamlit-folium \
    plotly \
    python-dotenv \
    requests \
    numpy \
    shapely \
    transformers \
    accelerate \
    bitsandbytes \
    scipy

# 4. Copy application code
COPY . .

# 5. Create log directory as specified in app.py
RUN mkdir -p logs

# 6. Expose the default Streamlit port
EXPOSE 8501

# 7. Define the entrypoint command
# Runs the streamlit app, accessible on all network interfaces.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]