import pytest
import sys
import os
import json
import geopandas as gpd
from shapely.geometry import Point
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geoai_framework import GeoAIReasoningEngine, TaskType, GeoprocessingStep
from geoprocessing_tools import GeoprocessingToolkit

# Create a fixture for the GeoAIReasoningEngine to reuse it across tests
@pytest.fixture(scope="module")
def geoai_engine():
    """Provides a GeoAIReasoningEngine instance for the test suite."""
    return GeoAIReasoningEngine(model_name="llama-3-8b")

class TestGeoAISystem:

    def test_query_analysis_flood_risk(self, geoai_engine):
        """Tests that the LLM correctly identifies a flood risk assessment task."""
        query = "I need to assess flood risk for areas near rivers in Mumbai."
        analysis = geoai_engine.analyze_query(query)
        
        assert analysis["intent"] == TaskType.FLOOD_RISK
        assert "rivers" in analysis["entities"]
        assert "water_bodies" in analysis["data_requirements"]

    def test_workflow_planning_buffer(self, geoai_engine):
        """Tests that the LLM can generate a simple buffer workflow."""
        query_analysis = {
            "intent": TaskType.BUFFER_ANALYSIS,
            "entities": ["points"],
            "data_requirements": ["points_layer"]
        }
        
        workflow = geoai_engine.plan_workflow(query_analysis)
        
        assert len(workflow) > 0
        assert workflow[0].operation == 'buffer'
        assert "distance" in workflow[0].parameters

    def test_end_to_end_workflow_execution(self, geoai_engine):
        """ 
        Tests a full workflow from query to execution, using a mock LLM response
        to ensure the geoprocessing tools are called correctly.
        """
        # Create a sample GeoDataFrame to act as the input data
        points = [Point(0, 0), Point(1, 1)]
        input_gdf = gpd.GeoDataFrame([1, 2], geometry=points, crs="EPSG:4326")
        data_sources = {"points_layer": input_gdf}

        # Define a simple workflow to test execution
        workflow_steps = [
            GeoprocessingStep(
                step_id="step_1",
                operation="buffer",
                parameters={"distance": 100, "input_layer": "points_layer"},
                input_data=["points_layer"],
                output_data="buffered_points",
                reasoning="Buffer the input points to create zones of interest."
            )
        ]

        # Execute the workflow
        result = geoai_engine.execute_workflow(workflow_steps, data_sources)

        # Verify the results
        assert result.success is True
        assert "buffered_points" in result.outputs
        assert isinstance(result.outputs["buffered_points"], gpd.GeoDataFrame)
        assert len(result.outputs["buffered_points"]) == 2

if __name__ == "__main__":
    pytest.main([__file__])

