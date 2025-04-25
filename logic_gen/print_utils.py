from datetime import datetime
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

# ANSI color codes for formatted output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    CODE_BLOCK = '\033[100m'  # Gray background for code blocks

def log_step(step: str, message: str, color: str = Colors.BLUE):
    """Print a formatted log message for a pipeline step."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {Colors.BOLD}{step}{Colors.ENDC}{color} → {message}{Colors.ENDC}")

def log_success(message: str):
    """Print a success message with GREEN color."""
    log_step("SUCCESS", message, Colors.GREEN)

def log_error(message: str):
    """Print an error message with RED color."""
    log_step("ERROR", message, Colors.RED)

def log_warning(message: str):
    """Print a warning message with YELLOW color."""
    log_step("WARNING", message, Colors.YELLOW)

def log_info(message: str):
    """Print an informational message with CYAN color."""
    log_step("INFO", message, Colors.CYAN)

class DisplayConfig:
    SHOW_CODE = True  # Toggle this to False to suppress code block display

def display_code(code: str, title: str = None):
    """
    Print a code string with syntax highlighting if DisplayConfig.SHOW_CODE is True.
    Optionally include a title for the code block.
    """
    if not DisplayConfig.SHOW_CODE:
        return
    # Print top border of code block
    print(f"\n{Colors.CODE_BLOCK}{'='*80}{Colors.ENDC}")
    if title:
        print(f"{Colors.BOLD}{Colors.BLUE}[CODE] {title}{Colors.ENDC}")
    print(f"{Colors.CODE_BLOCK}{'-'*80}{Colors.ENDC}")
    # Syntax-highlight the code and print
    highlighted = highlight(code, PythonLexer(), Terminal256Formatter(style='monokai'))
    print(highlighted, end="")
    # Print bottom border of code block
    print(f"{Colors.CODE_BLOCK}{'='*80}{Colors.ENDC}\n")
