import os
import json
import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import geopandas as gpd
import rasterio
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from config import Config
from geoprocessing_tools import GeoprocessingToolkit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    FLOOD_RISK = "flood_risk_assessment"
    SITE_SUITABILITY = "site_suitability_analysis"
    LAND_COVER = "land_cover_monitoring"
    BUFFER_ANALYSIS = "buffer_analysis"
    OVERLAY_ANALYSIS = "overlay_analysis"
    UNKNOWN = "unknown"

@dataclass
class GeoprocessingStep:
    """Represents a single step in a geoprocessing workflow"""
    step_id: str
    operation: str
    parameters: Dict[str, Any]
    input_data: List[str]
    output_data: str
    reasoning: str
    dependencies: List[str] = None

@dataclass
class WorkflowResult:
    """Contains the results of a completed workflow"""
    workflow_id: str
    steps: List[GeoprocessingStep]
    outputs: Dict[str, Any]
    reasoning_chain: List[str]
    execution_time: float
    success: bool
    error_messages: List[str] = None

class GeoAIReasoningEngine:
    """Chain-of-Thought reasoning engine for geospatial analysis, powered by an LLM."""
    
    def __init__(self, model_name: str = "mistral-7b"):
        self.model_config = Config.get_model_config(model_name)
        self.token = os.getenv("HUGGINGFACE_API_KEY")
        
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_path, token=self.token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_path, 
            token=self.token,
            quantization_config=quantization_config,
            device_map="auto" # Automatically handle device placement
        )

        self.geoprocessing_toolkit = GeoprocessingToolkit()
        self.reasoning_history = []
        logger.info(f"Initialized GeoAIReasoningEngine with model: {self.model_config.name}")

    def _invoke_llm(self, prompt: str) -> str:
        """Generates a response from the LLM and extracts a clean JSON string."""
        try:
            # Move inputs to the same device as the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Find the start of the first JSON object or array
            start_brace = response.find('{')
            start_bracket = response.find('[')

            if start_brace == -1 and start_bracket == -1:
                logger.error("No JSON object or array found in LLM response.")
                return "[]"

            if start_brace == -1:
                json_start_index = start_bracket
            elif start_bracket == -1:
                json_start_index = start_brace
            else:
                json_start_index = min(start_brace, start_bracket)

            json_string = response[json_start_index:]
            
            # Use the decoder to parse one valid JSON object and ignore trailing text
            decoder = json.JSONDecoder()
            try:
                obj, end_index = decoder.raw_decode(json_string)
                # Return the substring that is valid JSON
                return json_string[:end_index]
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from LLM response: {e}")
                logger.debug(f"Problematic JSON string for decoder: {json_string}")
                return "[]" # Return an empty list for workflow planning on error

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return "[]"

    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Uses the LLM to analyze the user query and extract key information.
        """
        prompt = f"""
        Analyze the following geospatial query and extract the intent, key entities, 
        and data requirements. Return the analysis as a JSON object.

        Query: "{user_query}"

        Respond with a JSON object with the following keys:
        - "intent": One of {', '.join([t.value for t in TaskType])}.
        - "entities": A list of important geographical or thematic features (e.g., "rivers", "elevation").
        - "spatial_extent": A string describing the geographic area of interest (e.g., "Maharashtra, India").
        - "data_requirements": A list of data types needed (e.g., "dem", "water_bodies").

        JSON Response:
        """
        
        self.reasoning_history.append("Analyzing user query with LLM...")
        llm_response = self._invoke_llm(prompt)
        
        try:
            analysis = json.loads(llm_response)
            analysis['intent'] = TaskType(analysis.get('intent', 'unknown'))
            self.reasoning_history.append(f"Query Analysis Complete: Intent is {analysis['intent'].value}")
            return analysis
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response for query analysis: {e}")
            self.reasoning_history.append("Error: Could not understand the query.")
            return {"intent": TaskType.UNKNOWN, "entities": [], "data_requirements": []}

    def plan_workflow(self, query_analysis: Dict[str, Any]) -> List[GeoprocessingStep]:
        """
        Generates a step-by-step workflow using the LLM based on the query analysis.
        """
        available_tools = json.dumps(self.geoprocessing_toolkit.get_tool_schemas(), indent=2)
        
        prompt = f"""
[SYSTEM]
This is a new, independent request. Your sole task is to generate a JSON array of geoprocessing steps based on the provided analysis and tools. Your entire output must be only the JSON array. Do not add any other text or repeat the analysis object.
[/SYSTEM]

[ANALYSIS]
{json.dumps({k: v.value if isinstance(v, Enum) else v for k, v in query_analysis.items()}, indent=2)}
[/ANALYSIS]

[AVAILABLE_TOOLS]
{available_tools}
[/AVAILABLE_TOOLS]

[WORKFLOW_PLAN_JSON]
"""
        
        self.reasoning_history.append("Planning workflow with LLM...")
        llm_response = self._invoke_llm(prompt)

        try:
            if not llm_response.strip():
                raise json.JSONDecodeError("Empty response from LLM", llm_response, 0)

            parsed_json = json.loads(llm_response)
            
            # Definitive check for the specific error mode.
            # If the model returns the analysis dict instead of a workflow list, it will have an 'intent' key.
            if isinstance(parsed_json, dict) and 'intent' in parsed_json:
                logger.error(
                    f"LLM returned the query analysis object instead of a workflow plan. "
                    f"Response: {parsed_json}"
                )
                return [] # Return an empty list to prevent crashing

            # Proceed if the response is a list (the expected format)
            if isinstance(parsed_json, list):
                workflow_steps = []
                for step_data in parsed_json:
                    if isinstance(step_data, dict) and 'step_id' in step_data and 'operation' in step_data:
                        workflow_steps.append(GeoprocessingStep(**step_data))
                    else:
                        logger.warning(f"Skipping malformed step in workflow plan: {step_data}")
                self.reasoning_history.append(f"Workflow Planned: {len(workflow_steps)} steps generated.")
                return workflow_steps
            else:
                logger.error(f"LLM response was not a list of steps. Response: {parsed_json}")
                return []

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to parse LLM response for workflow plan: {e}")
            self.reasoning_history.append("Error: Could not create a valid workflow plan.")
            return []

    def execute_workflow(self, steps: List[GeoprocessingStep], data_sources: Dict[str, Any]) -> WorkflowResult:
        """
        Executes the planned workflow steps using the GeoprocessingToolkit.
        """
        import time
        start_time = time.time()
        
        executed_steps = []
        outputs = data_sources.copy()
        errors = []
        
        try:
            for step in steps:
                # Check dependencies
                if step.dependencies:
                    for dep in step.dependencies:
                        if dep not in [s.step_id for s in executed_steps]:
                            raise Exception(f"Dependency {dep} not satisfied for {step.step_id}")
                
                # Prepare inputs for the current step
                step_inputs = {input_name: outputs[input_name] for input_name in step.input_data}

                # Execute step
                result = self._execute_step(step, step_inputs)
                
                if result.get("error"):
                    raise Exception(f"Step {step.step_id} ({step.operation}) failed: {result['error']}")

                outputs[step.output_data] = result['result']
                executed_steps.append(step)
                
                reasoning = f"Executed {step.operation}: {step.reasoning}"
                self.reasoning_history.append(reasoning)
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Workflow execution failed: {e}")
        
        execution_time = time.time() - start_time
        
        return WorkflowResult(
            workflow_id=f"workflow_{int(time.time())}",
            steps=executed_steps,
            outputs=outputs,
            reasoning_chain=self.reasoning_history.copy(),
            execution_time=execution_time,
            success=len(errors) == 0,
            error_messages=errors
        )
    
    def _execute_step(self, step: GeoprocessingStep, inputs: Dict[str, Any]) -> Any:
        """Executes a single step using the GeoprocessingToolkit."""
        return self.geoprocessing_toolkit.execute_operation(
            operation=step.operation,
            parameters=step.parameters,
            inputs=inputs
        )
    
    def generate_explanation(self, workflow_result: WorkflowResult) -> str:
        """Generate human-readable explanation of the workflow"""
        explanation = "## Geospatial Analysis Workflow Explanation\n\n"
        
        explanation += f"**Workflow ID:** {workflow_result.workflow_id}\n"
        explanation += f"**Execution Time:** {workflow_result.execution_time:.2f} seconds\n"
        explanation += f"**Status:** {'Success' if workflow_result.success else 'Failed'}\n\n"
        
        explanation += "### Chain of Thought Reasoning:\n"
        for i, reasoning in enumerate(workflow_result.reasoning_chain, 1):
            explanation += f"{i}. {reasoning}\n"
        
        explanation += "\n### Workflow Steps:\n"
        for step in workflow_result.steps:
            explanation += f"- **{step.operation}**: {step.reasoning}\n"
        
        if workflow_result.error_messages:
            explanation += "\n### Errors:\n"
            for error in workflow_result.error_messages:
                explanation += f"- {error}\n"
        
        return explanation

# Example usage
def main():
    """Example of how to use the GeoAI system"""
    
    # Initialize the reasoning engine
    geoai = GeoAIReasoningEngine(model_name="mistral-7b")
    
    # Example query
    user_query = "I need to assess flood risk for areas near rivers with elevation data in Maharashtra."
    
    # Analyze the query
    query_analysis = geoai.analyze_query(user_query)
    print("Query Analysis:", json.dumps(query_analysis, indent=2, default=str))
    
    # Plan workflow
    if query_analysis['intent'] != TaskType.UNKNOWN:
        workflow_steps = geoai.plan_workflow(query_analysis)
        print(f"\nPlanned {len(workflow_steps)} workflow steps")
        
        # Mock data sources for execution
        data_sources = {
            "elevation_raster": "path/to/dem.tif",
            "water_bodies_vector": "path/to/rivers.shp"
        }
        
        # Execute workflow
        # Note: This will fail until the geoprocessing tools are fully implemented
        # result = geoai.execute_workflow(workflow_steps, data_sources)
        
        # Generate explanation
        # explanation = geoai.generate_explanation(result)
        # print("\n" + explanation)

if __name__ == "__main__":
    main()
