# Use a community-maintained, multi-platform base image with GDAL pre-installed
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Install QGIS and other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gnupg \
    wget \
    python3-venv \
    && mkdir -p /etc/apt/keyrings \
    && wget -O /etc/apt/keyrings/qgis-archive-keyring.gpg https://download.qgis.org/downloads/qgis-archive-keyring.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/qgis-archive-keyring.gpg] https://qgis.org/ubuntu $(lsb_release -cs) main" >> /etc/apt/sources.list.d/qgis.list \
    && apt-get update \
    && apt-get install -y \
    python3-pip \
    python3-dev \
    qgis \
    python3-qgis \
    qgis-plugin-grass \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies into the virtual environment
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs cache logs temp

# Set permissions
RUN chmod +x run.sh

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
