import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from llm_sandbox import SandboxSession
from logic_gen.model_selection import tracer

@dataclass
class ExecutionResult:
    """Result of executing a script in the sandbox."""
    success: bool
    stdout: Optional[str] = None
    error: Optional[str] = None

def extract_code(response: str) -> (Optional[str], bool):
    """
    Extract the first Python code block from the given response string.
    Returns a tuple (code_string, success_flag).
    """
    code_blocks = re.findall(r'```python([\s\S]*?)```', response)
    if not code_blocks:
        return None, False
    code = code_blocks[0].strip()
    return code, True

def extract_code_with_retries(response: str, max_retries: int = 3) -> Optional[str]:
    """
    Try extracting a code block multiple times (if the model didn't format it correctly initially).
    Returns the code string if found, otherwise None.
    """
    for _ in range(max_retries):
        code, success = extract_code(response)
        if success:
            return code
    return None

def save_script(code: str, filename: str) -> Path:
    """Save the given code string to a file with the specified filename."""
    path = Path(filename)
    try:
        path.write_text(code)
    except Exception as e:
        raise RuntimeError(f"Error writing code to {filename}: {e}")
    return path

def execute_code(file: Path) -> ExecutionResult:
    """
    Execute a Python script file in a sandbox environment (Docker container).
    Automatically installs dependencies listed in the script's PEP 723 header (via `uv add`).
    Returns an ExecutionResult with stdout and a success flag.
    """
    try:
        with SandboxSession(image="python", keep_template=True, lang="python", verbose=True) as session:
            session.copy_to_runtime(file, "/app/sample.py")
            session.execute_command("uv add --script /app/sample.py pytest")
            result = session.execute_command("uv run /app/sample.py")
            output = result.text
            success = check_pytest_success(output)
            return ExecutionResult(success=success, stdout=output)
    except Exception as e:
        return ExecutionResult(success=False, error=str(e))

@tracer.chain
def execute_code_session(session: SandboxSession, file: Path) -> ExecutionResult:
    """
    Execute a Python script using an existing SandboxSession (to avoid reinitializing container each time).
    Assumes the session is already open. Installs dependencies and runs the script inside the sandbox.
    """
    try:
        session.execute_command("rm /app/sample.py")
        session.execute_command("rm /app/sample_updated.py")
        session.copy_to_runtime(file, "/app/sample.py")
        session.execute_command("uv add --script /app/sample.py pytest")
        result = session.execute_command("uv run /app/sample.py")
        output = result.text
        success = check_pytest_success(output)
        return ExecutionResult(success=success, stdout=output)
    except Exception as e:
        return ExecutionResult(success=False, error=str(e))

def check_pytest_success(stdout: str) -> bool:
    """
    Determine if the pytest output indicates all tests passed successfully.
    Returns True if no failures or errors are found, False otherwise.
    """
    if not stdout:
        return False
    # Convert to lowercase for uniform checking
    output = stdout.lower()
    if "failed" in output or "error" in output or "traceback" in output:
        # If any failure or error keyword is present in output, consider it a failure
        return False
    elif "passed" in output:
        return True
    else:
        False
