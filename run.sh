#!/bin/bash

# GeoAI System Startup Script

echo "🌍 Starting GeoAI Chain-of-Thought System..."

# Check if required directories exist
mkdir -p data outputs cache logs temp

# Check if environment file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Please copy .env.example to .env and configure your API keys."
    exit 1
fi

# Check if required Python packages are installed
python3 -c "import streamlit, geopandas, transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements.txt
fi

# Check GDAL installation
python3 -c "from osgeo import gdal" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ GDAL not properly installed. Please check your system dependencies."
    exit 1
fi

# Start the application
echo "🚀 Launching Streamlit application..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
