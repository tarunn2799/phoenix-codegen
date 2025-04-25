# Code Generation Framework


This framework automates the selection of optimal ML models for different tasks and generates production-ready code, significantly reducing the time and effort required for ML implementation. The project demonstrates this capability while comparing different agent-based approaches to code generation.

## Demo

If you have 5 minutes to listen to me yap about this project, here's a video:

[LOOM VIDEO](https://www.loom.com/share/d79bee5c18c14d9fbfa7ce34766347ad?sid=0fadd9cc-420d-4ede-9220-06538fd3f7e5)


But if you don't, please read this thorough project document:

[GOOGLE DOCS LINK](https://docs.google.com/document/d/1KtNk2uNBj83iN6jiA_adCo55AlxtS2t9I3Lfu1qsCoA/edit?usp=sharing)


## Features

- **Task-to-Code Generation**: Convert natural language task descriptions into working Python code
- **Automated Model Selection**: Intelligent selection of the best Hugging Face model for your specific NLP task
- **Test-Driven Development**: Automatic generation of test cases to validate the generated solution
- **Self-Healing Code**: Iterative feedback loop that refines the code until all tests pass
- **Sandbox Execution**: Secure code execution in an isolated Docker container

## Prerequisites

- Docker
- Python 3.11+
- API keys for LLM services (OpenAI, Anthropic)
- Hugging Face API token

## Installation

1. Unzip the repository:
   ```bash
   unzip final.zip
   cd final
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the `logic_gen` directory with your API keys (see the sample .env file below)

## Docker Setup

This project relies on a Docker container for secure code execution. Build the Docker image:

```bash
cd logic_gen
docker build -t sandboxpython .
```

## Usage

You can use the Logic Generator from the command line:

```bash
python main.py --task "Extract person names, locations, and dates from medical records"
```

Or import it into your own Python code:

```python
from logic_gen.logic_flow import generate_logic

# Generate logic for a specific task
result_file = generate_logic("Extract email addresses and phone numbers from customer support tickets")
if result_file:
    print(f"Logic generated successfully! Check {result_file}")
```

## How It Works

1. **Model Selection**: Analyzes your task to select the most appropriate Hugging Face model
2. **Initial Solution**: Generates an initial code solution using LLM-powered agents
3. **Test Generation**: Creates comprehensive test cases to validate the solution
4. **Feedback Loop**: Executes the code in a sandbox, analyzes errors, and iteratively improves the solution
5. **Final Output**: Returns working code that passes all tests

## Sample .env File

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
PHOENIX_API_KEY=your_phoenix_api_key
```

## Troubleshooting

- **Docker Issues**: Make sure Docker is running and the `sandboxpython` image has been built
- **API Key Errors**: Verify that all API keys in your `.env` file are valid
- **Model Loading Errors**: Check that you have a stable internet connection for downloading models

## License

[MIT License](LICENSE) 