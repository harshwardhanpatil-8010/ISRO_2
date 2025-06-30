import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from typing import Dict, List, Any
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

class GeoprocessingTool(ABC):
    """Abstract base class for geoprocessing tools."""

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Returns the JSON schema for the tool."""
        pass

    @abstractmethod
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the geoprocessing operation."""
        pass

class BufferTool(GeoprocessingTool):
    """Buffer analysis tool."""

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "buffer",
            "description": "Creates buffer zones around vector features.",
            "parameters": {
                "distance": {"type": "number", "description": "The buffer distance in the layer's units."},
                "input_layer": {"type": "string", "description": "Name of the input vector layer."}
            }
        }

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            distance = float(parameters.get('distance', 1000))
            input_gdf = inputs.get(parameters['input_layer'])
            
            if not isinstance(input_gdf, gpd.GeoDataFrame):
                return {'error': f"Input '{parameters['input_layer']}' is not a GeoDataFrame."}

            buffered_gdf = input_gdf.copy()
            buffered_gdf['geometry'] = input_gdf.geometry.buffer(distance)
            return {'result': buffered_gdf}
        except Exception as e:
            return {'error': str(e)}

class OverlayTool(GeoprocessingTool):
    """Spatial overlay operations."""

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "overlay",
            "description": "Performs spatial overlay between two vector layers.",
            "parameters": {
                "how": {"type": "string", "enum": ["intersection", "union", "difference"], "description": "The type of overlay to perform."},
                "layer1": {"type": "string", "description": "Name of the first input vector layer."},
                "layer2": {"type": "string", "description": "Name of the second input vector layer."}
            }
        }

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            how = parameters.get('how', 'intersection')
            gdf1 = inputs.get(parameters['layer1'])
            gdf2 = inputs.get(parameters['layer2'])

            if not all(isinstance(gdf, gpd.GeoDataFrame) for gdf in [gdf1, gdf2]):
                return {'error': "Both input layers must be GeoDataFrames."}

            result_gdf = gpd.overlay(gdf1, gdf2, how=how)
            return {'result': result_gdf}
        except Exception as e:
            return {'error': str(e)}

class SlopeAnalysisTool(GeoprocessingTool):
    """Slope analysis from elevation data."""

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "slope",
            "description": "Calculates slope from a raster elevation layer.",
            "parameters": {
                "input_raster": {"type": "string", "description": "Name of the input DEM raster layer."}
            }
        }

    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_raster_path = inputs.get(parameters['input_raster'])

            if not isinstance(input_raster_path, str) or not Path(input_raster_path).exists():
                return {'error': f"Input raster path is not a valid file: {input_raster_path}"}

            with rasterio.open(input_raster_path) as src:
                # A simple slope calculation (replace with a more robust method if needed)
                elevation = src.read(1)
                dx, dy = np.gradient(elevation, src.res[0], src.res[1])
                slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
                
                # Save the output raster
                output_path = f"./temp/slope_{Path(input_raster_path).stem}.tif"
                profile = src.profile
                profile.update(dtype=rasterio.float32, count=1, compress='lzw')

                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(slope.astype(rasterio.float32), 1)
                
                return {'result': output_path}
        except Exception as e:
            return {'error': str(e)}

class GeoprocessingToolkit:
    """Main toolkit class that orchestrates all geoprocessing operations."""
    
    def __init__(self):
        self.tools = {
            'buffer': BufferTool(),
            'overlay': OverlayTool(),
            'slope': SlopeAnalysisTool()
        }
        logger.info(f"GeoprocessingToolkit initialized with {len(self.tools)} tools.")
    
    def get_tool_schemas(self) -> Dict[str, Any]:
        """Returns a dictionary of all available tool schemas."""
        return {name: tool.get_schema() for name, tool in self.tools.items()}

    def execute_operation(self, operation: str, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a geoprocessing operation."""
        if operation not in self.tools:
            return {'error': f'Unknown operation: {operation}'}
        
        tool = self.tools[operation]
        
        try:
            result = tool.execute(parameters, inputs)
            logger.info(f"Successfully executed {operation}")
            return result
        except Exception as e:
            logger.error(f"Error executing {operation}: {str(e)}")
            return {'error': str(e)}