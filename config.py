import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str
    model_path: str
    max_tokens: int
    temperature: float
    top_p: float

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    api_url: str
    api_key: str
    rate_limit: int

class Config:
    """Main configuration class"""
    
    # Model configurations
    MODELS = {
        "gemma-2b-it": ModelConfig(
            name="Gemma-2B-IT",
            model_path="google/gemma-2b-it",
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9
        ),
        "mistral-7b": ModelConfig(
            name="Mistral-7B-Instruct",
            model_path="mistralai/Mistral-7B-Instruct-v0.2",
            max_tokens=4096,
            temperature=0.3,
            top_p=0.9
        ),
        "llama-3-8b": ModelConfig(
            name="LLaMA-3-8B",
            model_path="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=4096,
            temperature=0.3,
            top_p=0.9
        ),
        "phi-2": ModelConfig(
            name="Phi-2",
            model_path="microsoft/phi-2",
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9
        )
    }
    
    # Data source configurations
    DATA_SOURCES = {
        "bhoonidhi": DataSourceConfig(
            name="Bhoonidhi",
            api_url="https://bhoonidhi.nrsc.gov.in/api",
            api_key=os.getenv("BHOONIDHI_API_KEY", ""),
            rate_limit=100
        ),
        "osm": DataSourceConfig(
            name="OpenStreetMap",
            api_url="https://overpass-api.de/api/interpreter",
            api_key="",
            rate_limit=1000
        ),
        "stac": DataSourceConfig(
            name="STAC Catalog",
            api_url="https://earth-search.aws.element84.com/v1",
            api_key="",
            rate_limit=500
        )
    }
    
    # Processing configurations
    PROCESSING = {
        "max_concurrent_operations": 4,
        "timeout_seconds": 300,
        "cache_directory": "./cache",
        "output_directory": "./outputs",
        "temp_directory": "./temp"
    }
    
    # UI configurations
    UI = {
        "page_title": "GeoAI - Intelligent Spatial Analysis",
        "page_icon": "🌍",
        "layout": "wide",
        "max_query_length": 1000,
        "max_workflow_history": 50
    }
    
    # Coordinate systems
    COORDINATE_SYSTEMS = {
        "WGS84": "EPSG:4326",
        "Web Mercator": "EPSG:3857",
        "UTM 43N": "EPSG:32643",  # For Maharashtra, India
        "Indian 1975": "EPSG:24378"
    }
    
    # Geoprocessing tool configurations
    GEOPROCESSING_TOOLS = {
        "qgis": {
            "enabled": True,
            "path": "/usr/bin/qgis",
            "algorithms": [
                "native:buffer",
                "native:clip",
                "native:overlay",
                "gdal:slope",
                "qgis:distancematrix"
            ]
        },
        "gdal": {
            "enabled": True,
            "path": "/usr/bin/gdal",
            "utilities": [
                "gdalwarp",
                "gdal_calc",
                "gdal_polygonize",
                "gdal_rasterize",
                "gdal_translate"
            ]
        },
        "grass": {
            "enabled": True,
            "path": "/usr/bin/grass",
            "modules": [
                "r.slope.aspect",
                "r.buffer",
                "v.overlay",
                "r.mapcalc",
                "v.distance"
            ]
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get model configuration by name"""
        return cls.MODELS.get(model_name, cls.MODELS["mistral-7b"])
    
    @classmethod
    def get_data_source_config(cls, source_name: str) -> DataSourceConfig:
        """Get data source configuration by name"""
        return cls.DATA_SOURCES.get(source_name)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for dir_path in [
            cls.PROCESSING["cache_directory"],
            cls.PROCESSING["output_directory"],
            cls.PROCESSING["temp_directory"]
        ]:
            os.makedirs(dir_path, exist_ok=True)

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration settings"""
        errors = []
        
        # Check model configurations
        for model in cls.MODELS.values():
            if not model.name or not model.model_path:
                errors.append(f"Invalid model configuration: {model}")
        
        # Check data source configurations
        for source in cls.DATA_SOURCES.values():
            if not source.name or not source.api_url:
                errors.append(f"Invalid data source configuration: {source}")
        
        # Check processing directories
        for dir_path in [
            cls.PROCESSING["cache_directory"],
            cls.PROCESSING["output_directory"],
            cls.PROCESSING["temp_directory"]
        ]:
            if not os.path.exists(dir_path):
                errors.append(f"Directory does not exist: {dir_path}")
        
        return {"errors": errors} if errors else {"status": "valid"}