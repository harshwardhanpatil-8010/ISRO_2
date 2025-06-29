import streamlit_ui as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import folium
from streamlit_folium import st_folium

# Set page config
st.set_page_config(
    page_title="GeoAI - Intelligent Spatial Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #2a5298;
    }
    .reasoning-step {
        background: #e8f4f8;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #17a2b8;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow_history' not in st.session_state:
    st.session_state.workflow_history = []
if 'current_workflow' not in st.session_state:
    st.session_state.current_workflow = None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌍 GeoAI - Intelligent Spatial Analysis System</h1>
        <p>Chain-of-Thought Reasoning for Complex Geospatial Workflows</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ System Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Select LLM Model",
            ["Mistral-7B-Instruct", "LLaMA-3-8B", "Phi-2"],
            index=0
        )
        
        # Data sources
        st.subheader("📊 Data Sources")
        data_sources = st.multiselect(
            "Available Data Sources",
            ["Bhoonidhi", "OpenStreetMap", "SRTM DEM", "Landsat", "Sentinel-2"],
            default=["OpenStreetMap", "SRTM DEM"]
        )
        
        # Processing options
        st.subheader("⚙️ Processing Options")
        enable_rag = st.checkbox("Enable RAG (Retrieval-Augmented Generation)", value=True)
        show_reasoning = st.checkbox("Show Chain-of-Thought Reasoning", value=True)
        auto_execute = st.checkbox("Auto-execute workflow", value=False)
        
        # Workflow history
        st.subheader("📝 Workflow History")
        if st.session_state.workflow_history:
            for i, workflow in enumerate(st.session_state.workflow_history[-5:]):  # Show last 5
                if st.button(f"Load: {workflow['query'][:30]}...", key=f"history_{i}"):
                    st.session_state.current_workflow = workflow
                    st.rerun()
        else:
            st.info("No workflows executed yet")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💬 Natural Language Query")
        
        # Query input
        user_query = st.text_area(
            "Describe your geospatial analysis task:",
            placeholder="e.g., 'Assess flood risk for areas within 500m of rivers in Maharashtra with elevation below 100m'",
            height=100
        )
        
        # Quick examples
        st.subheader("📋 Example Queries")
        example_queries = [
            "Find suitable locations for solar farms avoiding forests and water bodies",
            "Assess flood risk near rivers using elevation and slope analysis", 
            "Identify urban expansion areas using land cover change detection",
            "Find optimal routes avoiding steep terrain and protected areas"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(example, key=f"example_{i}"):
                user_query = example
                st.rerun()
        
        # Process button
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            process_btn = st.button("🚀 Analyze Query", type="primary")
        with col_btn2:
            clear_btn = st.button("🗑️ Clear Results")
            if clear_btn:
                st.session_state.current_workflow = None
                st.rerun()
    
    with col2:
        st.header("🧠 Chain-of-Thought Reasoning")
        
        if process_btn and user_query:
            # Simulate workflow processing
            with st.spinner("Analyzing query and planning workflow..."):
                workflow_result = simulate_workflow_processing(user_query, model_choice, data_sources)
                st.session_state.current_workflow = workflow_result
                st.session_state.workflow_history.append(workflow_result)
        
        # Display reasoning steps
        if st.session_state.current_workflow and show_reasoning:
            display_reasoning_steps(st.session_state.current_workflow)
    
    # Results section
    if st.session_state.current_workflow:
        st.header("📊 Workflow Results")
        display_workflow_results(st.session_state.current_workflow)

def simulate_workflow_processing(query: str, model: str, data_sources: list) -> dict:
    """Simulate the workflow processing with realistic delays"""
    
    # Simulate query analysis
    time.sleep(1)
    
    # Determine task type
    query_lower = query.lower()
    if "flood" in query_lower:
        task_type = "Flood Risk Assessment"
        steps = [
            "Load elevation data (DEM)",
            "Calculate slope from elevation",
            "Identify water bodies from OSM",
            "Create buffer zones around rivers",
            "Perform overlay analysis",
            "Generate risk classification"
        ]
    elif "suitable" in query_lower or "optimal" in query_lower:
        task_type = "Site Suitability Analysis"
        steps = [
            "Define suitability criteria",
            "Load relevant spatial layers", 
            "Apply distance constraints",
            "Perform multi-criteria analysis",
            "Rank potential sites",
            "Generate suitability map"
        ]
    else:
        task_type = "General Spatial Analysis"
        steps = [
            "Parse spatial requirements",
            "Load required datasets",
            "Apply spatial operations",
            "Generate analysis results"
        ]
    
    # Simulate reasoning chain
    reasoning_chain = [
        f"Query Analysis: Identified {task_type} task",
        f"Data Requirements: {', '.join(data_sources)}",
        f"Model Selection: Using {model} for reasoning",
        f"Workflow Planning: Generated {len(steps)} processing steps",
        "Parameter Optimization: Adjusting for local coordinate system",
        "Execution Strategy: Sequential processing with error handling"
    ]
    
    return {
        "query": query,
        "task_type": task_type,
        "model": model,
        "data_sources": data_sources,
        "steps": steps,
        "reasoning_chain": reasoning_chain,
        "timestamp": datetime.now().isoformat(),
        "execution_time": 12.5,  # Simulated
        "success": True
    }

def display_reasoning_steps(workflow: dict):
    """Display the chain-of-thought reasoning steps"""
    
    for i, step in enumerate(workflow["reasoning_chain"], 1):
        st.markdown(f"""
        <div class="reasoning-step">
            <strong>Step {i}:</strong> {step}
        </div>
        """, unsafe_allow_html=True)

def display_workflow_results(workflow: dict):
    """Display comprehensive workflow results"""
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Task Type", workflow["task_type"])
    with col2:
        st.metric("Execution Time", f"{workflow['execution_time']}s")
    with col3:
        st.metric("Processing Steps", len(workflow["steps"]))
    with col4:
        status_color = "🟢" if workflow["success"] else "🔴"
        st.metric("Status", f"{status_color} {'Success' if workflow['success'] else 'Failed'}")
    
    # Tabs for different result views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Workflow Steps", "🗺️ Map Visualization", "📈 Performance Metrics", "💾 Export Results"])
    
    with tab1:
        st.subheader("Generated Workflow Steps")
        for i, step in enumerate(workflow["steps"], 1):
            st.markdown(f"""
            <div class="step-container">
                <strong>Step {i}:</strong> {step}
            </div>
            """, unsafe_allow_html=True)
        
        # Workflow as JSON
        if st.checkbox("Show Workflow JSON"):
            workflow_json = {
                "workflow_id": f"workflow_{int(time.time())}",
                "query": workflow["query"],
                "steps": [{"step_id": f"step_{i+1}", "operation": step} for i, step in enumerate(workflow["steps"])],
                "metadata": {
                    "model": workflow["model"],
                    "data_sources": workflow["data_sources"],
                    "timestamp": workflow["timestamp"]
                }
            }
            st.json(workflow_json)
    
    with tab2:
        st.subheader("Spatial Analysis Visualization")
        
        # Create a sample map (this would show actual results in a real system)
        sample_map = create_sample_map(workflow["task_type"])
        st_folium(sample_map, width=700, height=400)
        
        # Additional charts
        if workflow["task_type"] == "Flood Risk Assessment":
            create_flood_risk_charts()
        elif "Suitability" in workflow["task_type"]:
            create_suitability_charts()
    
    with tab3:
        st.subheader("Performance Analysis")
        
        # Create performance metrics
        create_performance_dashboard(workflow)
    
    with tab4:
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📄 Export Workflow (JSON)"):
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(workflow, indent=2),
                    file_name=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🗺️ Export Map Layers"):
                st.info("Map layers would be exported as GeoJSON/Shapefile")

def create_sample_map(task_type: str):
    """Create a sample map based on task type"""
    
    # Center on Maharashtra, India
    center_lat, center_lon = 19.7515, 75.7139
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    if task_type == "Flood Risk Assessment":
        # Add flood risk zones
        folium.CircleMarker(
            [19.0760, 72.8777], # Mumbai
            radius=20,
            popup="High Risk Zone - Mumbai",
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=0.6
        ).add_to(m)
        
        folium.CircleMarker(
            [18.5204, 73.8567], # Pune  
            radius=15,
            popup="Medium Risk Zone - Pune",
            color="orange",
            fill=True,
            fillColor="orange", 
            fillOpacity=0.6
        ).add_to(m)
    
    elif "Suitability" in task_type:
        # Add suitable locations
        folium.Marker(
            [19.8762, 75.3433],
            popup="Suitable Location 1",
            icon=folium.Icon(color="green", icon="star")
        ).add_to(m)
        
        folium.Marker(
            [20.5937, 78.9629], 
            popup="Suitable Location 2",
            icon=folium.Icon(color="green", icon="star")
        ).add_to(m)
    
    return m

def create_flood_risk_charts():
    """Create charts specific to flood risk analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        risk_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High', 'Very High'],
            'Area (sq km)': [1250, 800, 450, 200],
            'Population': [50000, 75000, 25000, 10000]
        })
        
        fig = px.bar(risk_data, x='Risk Level', y='Area (sq km)', 
                    title="Flood Risk Distribution by Area",
                    color='Risk Level',
                    color_discrete_map={'Low': 'green', 'Medium': 'yellow', 
                                      'High': 'orange', 'Very High': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Elevation profile
        elevation_data = pd.DataFrame({
            'Distance (km)': range(0, 100, 5),
            'Elevation (m)': [100 + 50 * (i/10) - (i/10)**2 for i in range(0, 20)]
        })
        
        fig = px.line(elevation_data, x='Distance (km)', y='Elevation (m)',
                     title="Elevation Profile Along Analysis Transect")
        st.plotly_chart(fig, use_container_width=True)

def create_suitability_charts():
    """Create charts for suitability analysis"""
    
    # Suitability scores
    suitability_data = pd.DataFrame({
        'Criteria': ['Distance to Roads', 'Slope', 'Land Use', 'Water Access', 'Soil Type'],
        'Weight': [0.25, 0.20, 0.25, 0.15, 0.15],
        'Score': [8.5, 7.2, 9.1, 6.8, 7.9]
    })
    
    fig = px.scatter(suitability_data, x='Weight', y='Score', size='Score',
                    text='Criteria', title="Multi-Criteria Suitability Analysis")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

def create_performance_dashboard(workflow: dict):
    """Create performance metrics dashboard"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time breakdown
        time_data = pd.DataFrame({
            'Operation': ['Query Analysis', 'Data Loading', 'Processing', 'Visualization'],
            'Time (seconds)': [1.2, 3.8, 6.5, 1.0]
        })
        
        fig = px.pie(time_data, values='Time (seconds)', names='Operation',
                    title="Processing Time Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Resource usage
        resource_data = pd.DataFrame({
            'Metric': ['CPU Usage', 'Memory Usage', 'Disk I/O', 'Network'],
            'Percentage': [65, 42, 28, 15]
        })
        
        fig = px.bar(resource_data, x='Metric', y='Percentage',
                    title="Resource Utilization")
        st.plotly_chart(fig, use_container_width=True)
    
    # System metrics table
    st.subheader("Detailed Metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Total Execution Time', 'Data Processing Time', 'Model Inference Time', 
                  'Memory Peak Usage', 'Cache Hit Rate', 'Error Rate'],
        'Value': [f'{workflow["execution_time"]}s', '8.3s', '2.1s', '1.2GB', '87%', '0%'],
        'Target': ['<15s', '<10s', '<5s', '<2GB', '>80%', '<5%'],
        'Status': ['✅ Good', '✅ Good', '✅ Good', '✅ Good', '✅ Good', '✅ Excellent']
    })
    st.dataframe(metrics_df)

if __name__ == "__main__":
    main()