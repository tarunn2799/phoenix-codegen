import os
import json
import requests
import re
from typing import Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv
from smolagents import CodeAgent, tool, LiteLLMModel
from logic_gen.print_utils import log_step, log_success, log_error, log_warning, log_info

load_dotenv()

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={os.environ.get('PHOENIX_API_KEY')}"
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.environ.get('PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
from phoenix.otel import register

# configure the Phoenix tracer
tracer_provider = register(
  project_name="arize-takehome", # Default is 'default'
  endpoint="https://app.phoenix.arize.com/v1/traces"
)

tracer = tracer_provider.get_tracer(__name__)
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize the LiteLLM model for agents
model = LiteLLMModel(
    model_id="anthropic/claude-3-7-sonnet-20250219",
    temperature=0.1,
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Set up authentication for Hugging Face API
huggingface_api_key = os.environ.get("HUGGINGFACE_TOKEN")
headers = {"Authorization": f"Bearer {huggingface_api_key}"}

# Tool to search for models on Hugging Face Hub
@tool
def search_huggingface_models(
    query: str, 
    task_filter: Optional[str] = None,
    sort_by: str = "downloads", 
    direction: str = "-1",
    limit: int = 5
) -> Dict:
    """
    Searches for models on the Hugging Face Hub based on a query.
    
    Args:
        query: The search query string to find relevant models
        task_filter: Optional filter for specific tasks (e.g., "text-classification")
        sort_by: Field to sort results by (default: "downloads")
        direction: Sort direction, -1 for descending, 1 for ascending
        limit: Maximum number of results to return
        
    Returns:
        Dictionary containing search results or error information
    """
    base_url = "https://huggingface.co/api/models"
    params = {
        "search": query,
        "sort": sort_by,
        "direction": direction,
        "limit": limit,
        "full": "true"
    }
    
    if task_filter and task_filter != "":
        params["filter"] = task_filter
    
    try:
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        return {"success": True, "results": response.json(), "count": len(response.json())}
    except Exception as e:
        return {"success": False, "error": str(e), "count": 0, "results": []}

# Tool to get detailed information about a specific model
@tool
def get_model_details(model_id: str) -> Dict:
    """
    Gets detailed information about a specific model from the Hugging Face Hub.
    
    Args:
        model_id: The identifier of the model on Hugging Face Hub
        
    Returns:
        Dictionary containing model details or error information
    """
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {"success": True, "details": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e), "details": {}}

# Tool to fetch model card content
@tool
def fetch_model_card(model_id: str) -> str:
    """
    Fetches the README.md content (model card) for a specific model.
    
    Args:
        model_id: The identifier of the model on Hugging Face Hub
        
    Returns:
        String containing the model card markdown content or empty string if not found
    """
    url = f"https://huggingface.co/{model_id}/raw/main/README.md"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return ""
    except Exception as e:
        return ""

# Tool to extract code examples from model card
@tool
def extract_code_examples(model_card_md: str) -> List[str]:
    """
    Extracts code examples from a model card markdown.
    
    Args:
        model_card_md: The markdown content of the model card
        
    Returns:
        List of strings containing code examples extracted from the markdown
    """
    if isinstance(model_card_md, dict):
        model_card_md = model_card_md.get("content", "")

    # Find code blocks (```python ... ```)
    code_block_pattern = r"```(?:python|bash)?\s*([\s\S]*?)```"
    code_blocks = re.findall(code_block_pattern, model_card_md)
    
    # Clean up code blocks
    cleaned_examples = []
    for block in code_blocks:
        # Skip very short blocks that are likely not complete examples
        if len(block.strip().split("\n")) > 2:
            cleaned_examples.append(block.strip())
    
    return cleaned_examples

# Tool to format model summary
@tool
def format_model_summary(model_data: Dict, model_card: str = "", code_examples: List[str] = []) -> str:
    """
    Formats a comprehensive summary of a model.
    
    Args:
        model_data: Dictionary containing model metadata
        model_card: String containing the model card markdown content
        code_examples: List of code examples extracted from the model card
        
    Returns:
        Formatted string containing a comprehensive model summary
    """
    if not model_data:
        return "No suitable model was found."
    
    summary_parts = []
    
    # Model name and author
    summary_parts.append(f"# {model_data.get('modelId', 'Unknown Model')}")
    author = model_data.get('author', 'Unknown Author')
    summary_parts.append(f"**Author:** {author}")
    
    # Model link
    model_id = model_data.get('modelId', '')
    if model_id:
        summary_parts.append(f"**Model Link:** [https://huggingface.co/{model_id}](https://huggingface.co/{model_id})\n")
    
    # Downloads and likes
    downloads = model_data.get('downloads', 0)
    likes = model_data.get('likes', 0)
    summary_parts.append(f"**Downloads:** {downloads:,} | **Likes:** {likes}")
    
    # Last modified
    last_modified = model_data.get('lastModified', '')
    if last_modified:
        summary_parts.append(f"**Last Updated:** {last_modified.split('T')[0]}\n")
    
    # Tags
    tags = model_data.get('tags', [])
    if tags:
        summary_parts.append("**Tags:**")
        summary_parts.append(", ".join(tags))
        summary_parts.append("")
    
    # Description
    description = model_data.get('description', 'No description available')
    summary_parts.append("## Description")
    summary_parts.append(description)
    summary_parts.append("")
    
    # Usage examples
    summary_parts.append("## Usage Instructions")
    
    pipeline_tag = next((tag for tag in tags if tag.startswith("pipeline_tag:")), None)
    if pipeline_tag:
        pipeline_type = pipeline_tag.replace("pipeline_tag:", "")
        summary_parts.append(f"This model works with the `{pipeline_type}` pipeline in Hugging Face Transformers.")
    
    if code_examples:
        summary_parts.append("### Code Examples")
        summary_parts.append("```python")
        for example in code_examples:
            if "@" not in example and "task" not in example:
                summary_parts.append(example)
        summary_parts.append("```")
    else:
        model_type = next((tag for tag in tags if tag.startswith("model-type:")), "")
        if model_type:
            model_type = model_type.replace("model-type:", "")
            summary_parts.append(f"This model is of type `{model_type}`.")
            
        # Generic usage example based on task type
        if any(tag in tags for tag in ["token-classification", "ner"]):
            summary_parts.append("```python")
            summary_parts.append(f"""from transformers import pipeline

# Load NER pipeline
ner_pipeline = pipeline("token-classification", model="{model_id}")

# Example text
text = "The patient was prescribed Aspirin and Lisinopril for treatment."

# Extract entities
entities = ner_pipeline(text)
print(entities)
""")
            summary_parts.append("```")
    
    return "\n".join(summary_parts)

# Create the model selector agent
def create_model_picker_agent():
    """Create an agent for finding the best model"""
    agent = CodeAgent(
        tools=[
            search_huggingface_models, 
            # get_model_details
        ],
        model=model,
        additional_authorized_imports=["json", "requests", "re"],
        max_steps=5
    )
    
    return agent

def create_model_details_agent():
    """Create an agent for getting detailed model information"""
    agent = CodeAgent(
        tools=[
            get_model_details,
            fetch_model_card,
            extract_code_examples,
            format_model_summary
        ],
        model=model,
        additional_authorized_imports=["json", "requests", "re"],
        max_steps=5
    )
    
    return agent

def get_model_labels(model_id: str) -> List[str]:
    """Get the labels for a specific model."""
    log_info(f"Fetching labels for model: {model_id}")
    try:
        url = f"https://huggingface.co/{model_id}/raw/main/config.json"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            config = response.json()
            if "label2id" in config:
                return list(config["label2id"].keys())
            if "id2label" in config:
                return list(config["id2label"].values())
        
        # If labels couldn't be found in config, try to get model details
        model_details = get_model_details(model_id)
        if model_details.get("success") and model_details.get("details"):
            tags = model_details["details"].get("tags", [])
            # Check for tags that might indicate entity types
            entity_tags = [tag for tag in tags if "entity" in tag.lower() or "ner" in tag.lower()]
            if entity_tags:
                return entity_tags
        return []
    except Exception as e:
        log_error(f"Error fetching labels for {model_id}: {e}")
        return []

def select_best_model(user_prompt: str) -> Tuple[str, List[str], str, str]:
    """
    Select the best model for a given user prompt using agent-based approach.
    
    Args:
        user_prompt: The user's prompt describing the NER task
        
    Returns:
        tuple: (model_id, labels, explanation, usage_example)
    """
    log_step("MODEL SELECTION", "Finding the best model using agent-based approach")
    
    # First agent to pick the best model
    picker_agent = create_model_picker_agent()
    
    log_info("Finding the best Hugging Face model...")
    picker_result = picker_agent.run(
        f"""
        You are a helpful assistant that finds the best Hugging Face model for this use case: "{user_prompt}"
        
        Follow these steps:
        1. First, analyze the query to determine the appropriate task type(s) for this use case:
           - Think step-by-step about which NLP task(s) would best address the user's needs
           - Consider Hugging Face's standard task categories like: "token-classification", "named-entity-recognition", 
             "text-classification", "sentiment-analysis", "summarization", "question-answering", "translation", 
             "text-generation", "text2text-generation"
           - For domain-specific needs, include domain filter terms like "biomedical" or "healthcare"
           - Determine 1-3 most appropriate task filters for the query
           - For the search query, use only the relevant terms. eg. "I want to remove all PII from my text" should be "pii" since pii models exist.
           - Use only the most unique term in the query, and filter by the task.
        
        2. Call search_huggingface_models with the most relevant task filter first (most specific)
        3. If results are insufficient, try the second most relevant task filter
        4. If still insufficient, try a keyword search without task filters
        5. Analyze the top results to find the best model based on:
           - Relevance to the specific task
           - Downloads and community adoption
           - Specificity for the domain (e.g., medical/drug-related)
           - For extraction tasks, ensure the model contains a label for the entity you want to extract.
        
        Return only the model_id of the best model you found. No explanation needed, just the model_id as a string.
        """
    )
    
    # Extract model_id from the first agent's result
    if isinstance(picker_result, dict):
        model_id = picker_result.get("model_id", str(picker_result))
    else:
        model_id = str(picker_result).strip()
    
    if not model_id:
        log_warning("No model ID found, using fallback model")
        model_id = "urchade/gliner_medium-v2.1"
    
    log_success(f"Selected model: {model_id}")
    
    # Second agent to get detailed information about the selected model
    details_agent = create_model_details_agent()
    
    log_info("Getting detailed information about the selected model...")
    details_result = details_agent.run(
        f"""
        You are a helpful assistant that provides detailed information about the Hugging Face model: "{model_id}"
        I am a developer who will use your model summary to build an application. I am specifically looking for python code usage instructions for this model.
        
        Follow these steps:
        1. Fetch detailed information about the model using get_model_details
        2. Fetch the model card (README.md) using fetch_model_card. Read the entire model card, dont truncate it. 
        3. Extract code examples from the model using extract_code_examples
        4. Format a comprehensive summary with all relevant details using format_model_summary
        
        Ensure the summary contains:
        - The model name, author, model link
        - Downloads, likes, last updated
        - Description
        - Usage examples and code examples (in python)
        
        For code examples:
        - Add these examples to the final output
        - In your output, always wrap Python examples in a single code block using ```python markdown syntax
        - Do not execute the code examples, just provide what you find

        Return the complete formatted summary.
        """
    )
    
    # Extract model labels and usage example from the result
    try:
        # Extract code examples
        code_blocks = re.findall(r"```python\s*([\s\S]*?)```", str(details_result))
        if code_blocks:
            usage_example = code_blocks[0].strip()
        else:
            # Generate a default usage example
            usage_example = f"""from transformers import pipeline

# Load NER pipeline
ner = pipeline("token-classification", model="{model_id}")

# Example text
text = "Example text to identify named entities."

# Extract entities
entities = ner(text)
print(entities)"""
        
        # Get model labels
        labels = get_model_labels(model_id)
        
        # Create explanation
        explanation = f"Selected model {model_id} based on task requirements."
    
    except Exception as e:
        log_error(f"Error processing model details: {e}")
        usage_example = f"from transformers import pipeline\nner = pipeline('token-classification', model='{model_id}')\nresult = ner('Example text')"
        labels = []
        explanation = f"Selected model {model_id} (error processing details: {e})"
    
    log_success(f"Best model determined: {model_id}")
    return model_id, labels, explanation, usage_example

if __name__ == "__main__":
    # Example usage of the model_selection module
    example_prompt = "Extract person names and organizations from news articles"
    result_id, result_labels, explanation, usage = select_best_model(example_prompt)
    print(f"Best suited model for the task: {result_id}")
    print(f"Labels for the best model: {result_labels}")
    print(f"Explanation: {explanation}")
    print(f"Model Usage: {usage}")
