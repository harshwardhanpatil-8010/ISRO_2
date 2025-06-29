import streamlit as st
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import login

# Import custom modules
from geoai_framework import GeoAIReasoningEngine, TaskType, WorkflowResult
from config import Config
from data_sources import DataSourceManager, DataQuery
from geoprocessing_tools import GeoprocessingToolkit

os.makedirs("logs", exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/geoai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title=Config.UI["page_title"],
    page_icon=Config.UI["page_icon"],
    layout=Config.UI["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .reasoning-step {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        animation: slideIn 0.5s ease-out;
    }
    .workflow-step {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class GeoAIApp:
    def __init__(self):
        self.initialize_session_state()
        self.geoai_engine, self.data_manager, self.geoprocessing_toolkit = self.load_system_components()

    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        if 'current_workflow' not in st.session_state:
            st.session_state.current_workflow = None
    
    @st.cache_resource
    def load_system_components(self):
        try:
            from huggingface_hub import login
            hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            login(token=hf_token)

            engine = GeoAIReasoningEngine()
            data_manager = DataSourceManager()
            toolkit = GeoprocessingToolkit()
            logger.info("System components loaded successfully")
            return engine, data_manager, toolkit
        except Exception as e:
            st.error(f"Failed to load system components: {str(e)}")
            logger.error(f"Component loading error: {e}")
            return None, None, None


    def render_header(self):
        """Render the main application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🌍 GeoAI - Intelligent Spatial Analysis System</h1>
            <p>Chain-of-Thought Reasoning for Complex Geospatial Workflows</p>
            <p><em>Powered by Advanced Language Models & Geospatial Intelligence</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("🛠️ System Configuration")
            
            model_options = list(Config.MODELS.keys())
            selected_model = st.selectbox("Select LLM Model", model_options, index=model_options.index("gemma-2b-it"))
            
            st.subheader("📊 Data Sources")
            st.info("Data sources are now determined automatically by the AI based on your query.")

            st.subheader("⚙️ Processing Options")
            show_reasoning = st.checkbox("Show Chain-of-Thought", value=True)
            
            st.subheader("🗺️ Area of Interest (Optional)")
            bbox_input = st.text_input("Bounding Box (min_lon,min_lat,max_lon,max_lat)", "72.8,18.9,73.0,19.2")

            return {
                'model': selected_model,
                'show_reasoning': show_reasoning,
                'bbox': tuple(map(float, bbox_input.split(','))) if bbox_input else None
            }
    
    def render_query_interface(self):
        """Render the query input interface"""
        st.header("💬 Natural Language Query Interface")
        
        example_queries = [
            "Assess flood risk for urban areas within 1km of rivers in Mumbai using elevation data",
            "Find suitable locations for solar farms in Maharashtra avoiding forests and water bodies",
        ]
        
        with st.expander("📋 Example Queries"):
            for i, example in enumerate(example_queries):
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    st.session_state.user_query = example
                    st.rerun()
        
        user_query = st.text_area(
            "Describe your geospatial analysis task:",
            value=st.session_state.get('user_query', ''),
            placeholder="e.g., 'Find optimal locations for wind farms in coastal areas with wind speed > 7 m/s'",
            height=120
        )
        return user_query

    def fetch_data_for_workflow(self, data_requirements: List[str], bbox: Optional[tuple]) -> Dict[str, Any]:
        """Fetches the necessary data for a workflow using the DataSourceManager."""
        fetched_data = {}
        if not bbox:
            st.warning("No area of interest defined. Using default bounding box for Mumbai.")
            bbox = (72.8, 18.9, 73.0, 19.2) # Default BBox for Mumbai

        for req in data_requirements:
            with st.spinner(f"Fetching {req} data..."):
                # This logic can be expanded to be more intelligent
                source_map = {
                    "water_bodies": ("osm", {"feature_type": "water"}),
                    "roads": ("osm", {"feature_type": "roads"}),
                    "buildings": ("osm", {"feature_type": "buildings"}),
                    "dem": ("bhoonidhi", {"product": "SRTM 1-arc-second DEM"}) # Placeholder
                }
                
                if req in source_map:
                    source_name, filters = source_map[req]
                    query = DataQuery(query_id=f"query_{req}", bbox=bbox, filters=filters)
                    result = self.data_manager.fetch_data(source_name, query)
                    
                    if result.success:
                        fetched_data[req] = result.data
                        st.success(f"Successfully fetched {req}.")
                    else:
                        st.error(f"Failed to fetch {req}: {result.error_message}")
                else:
                    st.warning(f"No data source configured for requirement: {req}")
        return fetched_data

    def process_query(self, query: str, config: Dict):
        """Process the user query, plan, and execute the workflow."""
        if not query.strip():
            st.warning("Please enter a query to analyze.")
            return

        with st.spinner("Analyzing query..."):
            query_analysis = self.geoai_engine.analyze_query(query)
        
        if query_analysis['intent'] == TaskType.UNKNOWN:
            st.error("Could not understand the geospatial intent of the query.")
            return

        with st.spinner("Planning workflow..."):
            workflow_steps = self.geoai_engine.plan_workflow(query_analysis)

        if not workflow_steps:
            st.error("Failed to generate a valid workflow plan.")
            return

        initial_data = self.fetch_data_for_workflow(query_analysis.get('data_requirements', []), config['bbox'])

        with st.spinner("Executing workflow..."):
            workflow_result = self.geoai_engine.execute_workflow(workflow_steps, initial_data)
        
        st.session_state.current_workflow = workflow_result
        st.session_state.workflow_history.append(workflow_result)

    def render_reasoning_display(self, workflow_result: WorkflowResult):
        """Render the chain-of-thought reasoning display"""
        st.header("🧠 Chain-of-Thought Reasoning")
        if workflow_result and workflow_result.reasoning_chain:
            for i, reasoning in enumerate(workflow_result.reasoning_chain, 1):
                st.markdown(f'''<div class="reasoning-step"><strong>Step {i}:</strong> {reasoning}</div>''', unsafe_allow_html=True)
        else:
            st.info("No reasoning steps available.")

    def render_workflow_steps(self, workflow_result: WorkflowResult):
        """Render the workflow steps visualization"""
        st.header("📋 Generated Workflow")
        if workflow_result and workflow_result.steps:
            for i, step in enumerate(workflow_result.steps, 1):
                st.markdown(f'''
                <div class="workflow-step">
                    <h4>Step {i}: {step.operation}</h4>
                    <p><strong>Reasoning:</strong> {step.reasoning}</p>
                    <p><strong>Parameters:</strong> <code>{json.dumps(step.parameters, indent=2)}</code></p>
                    <p><strong>Input:</strong> {step.input_data}</p>
                    <p><strong>Output:</strong> {step.output_data}</p>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No workflow steps generated.")

    def render_results_visualization(self, workflow_result: WorkflowResult):
        """Render results visualization dynamically based on workflow output."""
        st.header("📊 Analysis Results")
        if not workflow_result or not workflow_result.success:
            st.warning("Workflow did not complete successfully. No results to display.")
            if workflow_result and workflow_result.error_messages:
                st.error(f"Error: {workflow_result.error_messages[0]}")
            return

        # Metrics, etc.

        tab1, tab2, tab3 = st.tabs(["🗺️ Map View", "📋 Data Table", "💾 Export"])
        
        with tab1:
            self.render_map_visualization(workflow_result)
        with tab2:
            self.render_data_table(workflow_result)
        with tab3:
            self.render_export_options(workflow_result)

    def render_map_visualization(self, workflow_result: WorkflowResult):
        """Renders a map visualizing the output GeoDataFrames from the workflow."""
        st.subheader("Map Visualization")
        
        # Find the first GeoDataFrame in the outputs to determine the center
        map_center = None
        for output in workflow_result.outputs.values():
            if isinstance(output, gpd.GeoDataFrame) and not output.empty:
                bounds = output.total_bounds
                map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                break
        
        m = folium.Map(location=map_center if map_center else [19.07, 72.87], zoom_start=12)

        # Add all GeoDataFrame outputs to the map
        for name, output in workflow_result.outputs.items():
            if isinstance(output, gpd.GeoDataFrame):
                st.write(f"Visualizing layer: **{name}** ({len(output)} features)")
                folium.GeoJson(output, name=name).add_to(m)
        
        folium.LayerControl().add_to(m)
        st_folium(m, width=700, height=500)

    def render_data_table(self, workflow_result: WorkflowResult):
        """Renders a data table for the first output GeoDataFrame."""
        st.subheader("Data Preview")
        for name, output in workflow_result.outputs.items():
            if isinstance(output, gpd.GeoDataFrame):
                st.write(f"Data from layer: **{name}**")
                st.dataframe(output.drop(columns='geometry', errors='ignore'))
                return # Show only the first one
        st.info("No tabular data found in workflow outputs.")

    def render_export_options(self, workflow_result: WorkflowResult):
        """Render export options for workflow outputs."""
        st.subheader("Export Results")
        for name, output in workflow_result.outputs.items():
            if isinstance(output, gpd.GeoDataFrame):
                st.download_button(
                    label=f"📥 Download {name} as GeoJSON",
                    data=output.to_json(),
                    file_name=f"{name}.geojson",
                    mime="application/geo+json"
                )

    def run(self):
        """Main application runner"""
        self.render_header()
        self.geoai_engine, self.data_manager, self.geoprocessing_toolkit = self.load_system_components()

        if not self.geoai_engine:
            return

        config = self.render_sidebar()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            user_query = self.render_query_interface()
            if st.button("🚀 Analyze Query", type="primary", use_container_width=True):
                self.process_query(user_query, config)
                st.rerun()
        
        with col2:
            if st.session_state.current_workflow and config['show_reasoning']:
                self.render_reasoning_display(st.session_state.current_workflow)
        
        if st.session_state.current_workflow:
            st.markdown("---")
            self.render_workflow_steps(st.session_state.current_workflow)
            st.markdown("---")
            self.render_results_visualization(st.session_state.current_workflow)


if __name__ == "__main__":
    app = GeoAIApp()
    app.run()