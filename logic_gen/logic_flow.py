import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import litellm  # LLM interface (e.g., OpenAI/Anthropic unified)
from litellm import completion

from logic_gen.model_selection import select_best_model
from logic_gen.agents import Agent
from logic_gen.prompts import MLE, TESTER
from logic_gen.print_utils import log_step, log_info, log_success, log_warning, log_error, display_code, Colors
from logic_gen.utils import extract_code_with_retries, save_script, execute_code_session


# Constants for retry limits
MAX_EXTRACT_RETRIES = 3
MAX_FEEDBACK_RETRIES = 5

@dataclass
class ExecutionResult:
    """Dataclass to hold the result of code execution."""
    success: bool
    stdout: Optional[str] = None
    error: Optional[str] = None

def run_mle_feedback_loop(tester_agent: Agent, initial_code: str) -> Tuple[bool, Optional[Path]]:
    """
    Run an iterative feedback loop: execute the code with tests, and use the tester agent 
    to generate code fixes based on test failures until success or retry limit reached.
    """
    current_code = initial_code
    updated_filename = "sample.py"
    log_info(f"Starting MLE feedback loop with maximum {MAX_FEEDBACK_RETRIES} attempts")
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    
    # Open a sandbox session once for repeated use
    from llm_sandbox import SandboxSession
    session = SandboxSession(image="sandboxpython", keep_template=True, lang="python", verbose=True)
    session.open()

    # Install the uv tool inside the sandbox so that it can be used for test commands
    log_info("Installing uv tool in the sandbox container")
    install_result = session.execute_command("pip install uv")
    if install_result.text:
        log_info("uv installation output: " + install_result.text)
    
    try:
        for attempt in range(MAX_FEEDBACK_RETRIES):
            log_step(f"ATTEMPT {attempt + 1}/{MAX_FEEDBACK_RETRIES}", "Executing current code version")
            # Save the current code to file and execute it in the sandbox
            code_file = save_script(current_code, updated_filename)
            log_info(f"Saved updated code to {updated_filename}")
            time.sleep(0.5)  # brief pause before execution
            
            result = execute_code_session(session, code_file)
            if result.success:
                log_success(f"Code execution successful on attempt {attempt + 1}!")
                display_code(current_code, "Final Successful Implementation")
                print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
                return True, code_file
            
            # Print out the detailed sandbox logs
            log_warning(f"Execution failed on attempt {attempt + 1}")
            if result.stdout:
                log_warning("Sandbox STDOUT:")
                print(result.stdout)
            if result.error:
                log_warning("Sandbox STDERR:")
                print(result.error)
            
            log_info("Getting MLE feedback and generating improved code...")
            feedback_content = result.stdout if result.stdout else result.error
            feedback_content = feedback_content or "No output available."
            
            # Use the tester agent with its feedback prompt to get a corrected code
            feedback_response = tester_agent.llm_call(prompt=feedback_content, feedback=True)
            new_code = extract_code_with_retries(feedback_response[-1]['content'])
            if not new_code:
                log_error(f"Failed to extract code from feedback on attempt {attempt + 1}")
                continue  # Try again in next iteration if available
            
            current_code = new_code
            log_info("Successfully generated a new code version")
            display_code(current_code, f"Updated Implementation (Attempt {attempt + 1})")
            print(f"\n{Colors.BOLD}{'-'*80}{Colors.ENDC}\n")
    finally:
        session.close()
    
    log_error(f"Failed to generate working code after {MAX_FEEDBACK_RETRIES} attempts")
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}\n")
    return False, None

def run_pipeline(user_prompt: str) -> Optional[Path]:
    """
    Execute the end-to-end NER pipeline for a given user prompt:
    1. Select the best NER model for the task.
    2. Generate initial solution code using the MLE agent.
    3. Generate test cases using the Tester agent.
    4. Run the feedback loop to refine the code until tests pass.
    Returns the Path of the final working code file, or None if failed.
    """
    # Step 1: Model selection (topic inference and best model determination)
    log_step("MODEL SELECTION", "Selecting best model for the task")
    
    try:
        # Use the agent-based model selection
        best_model_id, best_model_labels, explanation, model_usage = select_best_model(user_prompt)
        log_info(f"Best model selected: {best_model_id}")
        
        # Prepare context about the selected model to guide code generation
        model_context = (f"Use the Hugging Face model {best_model_id} with the following usage instructions:\n\n{model_usage}")
        if best_model_labels:
            model_context += f"\n\nThis model supports the labels: {', '.join(best_model_labels)}."
    except Exception as e:
        log_error(f"Error in model selection: {e}")
        # Fallback to a default model if selection fails
        best_model_id = "urchade/gliner_medium-v2.1"
        model_usage = """from transformers import pipeline

# Load NER pipeline
ner = pipeline("token-classification", model="urchade/gliner_medium-v2.1")

# Example text
text = "Example text for named entity recognition."

# Extract entities
entities = ner(text)
print(entities)"""
        model_context = (f"Use the Hugging Face model {best_model_id} with the following usage instructions:\n\n{model_usage}")
    
    # Initialize agents for code generation (MLE) and test generation/fix (Tester)
    mle_agent = Agent(sys_prompt=MLE['system_prompt'], model="gpt-4o")
    tester_agent = Agent(sys_prompt=TESTER['system_prompt'], feedback_prompt=TESTER['feedback_prompt'], model="gpt-4o")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    log_step("PIPELINE START", "Initializing MLE-Tester workflow")
    print(f"{'='*80}\n")
    
    # Step 2: Get initial solution code from MLE agent
    log_info("Getting initial solution from MLE")
    mle_response = mle_agent.llm_call(prompt=MLE['user_prompt'].format(task=user_prompt, model_context=model_context))
    initial_code = extract_code_with_retries(mle_response[-1]['content'])
    if not initial_code:
        log_error("Failed to extract initial solution code from MLE")
        return None
    code_file = save_script(initial_code, "sample_logic.py")
    log_success(f"Initial solution code saved to {code_file}")
    display_code(initial_code, "Initial MLE Solution")
    
    # Step 3: Generate test cases using Tester agent
    log_step("TEST GENERATION", "Generating test cases with Tester")
    tester_response = tester_agent.llm_call(prompt=TESTER['user_prompt'].format(python_code=initial_code))
    test_code = extract_code_with_retries(tester_response[-1]['content'])
    if not test_code:
        log_error("Failed to extract test code from Tester")
        return None
    test_file = save_script(test_code, "sample.py")
    log_success(f"Test code saved to {test_file}")
    display_code(test_code, "Generated Test Cases")
    
    # Step 4: Execute feedback loop to refine code until tests pass
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    log_step("FEEDBACK LOOP", "Starting execution and feedback cycle")
    print(f"{'='*80}{Colors.ENDC}\n")
    success, final_file = run_mle_feedback_loop(tester_agent, initial_code)
    
    if success:
        log_success("Pipeline completed successfully!")
        log_info(f"Final working code saved to: {final_file}")
        return final_file
    else:
        log_error(f"Pipeline failed after {MAX_FEEDBACK_RETRIES} feedback attempts")
        return None

def generate_logic(task_description: str) -> Optional[Path]:
    """
    Generate logic for a given task description.
    This function is meant to be called from external modules.
    
    Args:
        task_description: Description of the task to generate logic for
        
    Returns:
        Path to the generated logic file, or None if generation failed
    """
    return run_pipeline(task_description)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the NER pipeline on a given prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt describing the NER task")
    args = parser.parse_args()
    result_path = run_pipeline(args.prompt)
    if result_path:
        print(f"Pipeline finished successfully. Final code is saved in: {result_path}")
    else:
        print("Pipeline did not produce a successful result.")
