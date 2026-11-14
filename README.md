# LLM Evaluation Test with DeepEval and Ollama 3.1

This project evaluates LLM tool-calling behavior for a weather agent and validates responses using DeepEval.

## Overview

- Weather tool call validation
- LLM evaluation with DeepEval
- Local inference with Ollama
- Optional remote inference with xAI / Grok models

> Note: The weather tool only provides current conditions. It cannot answer historical weather questions such as weather over the last seven days.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Environment Variables

Add the following to your `.env` file:

```env
GROQ_API_KEY="your-groq-api-key"
WEATHER_API_KEY="your-openweathermap-api-key"
```

The Groq agent script uses Groq's `qwen/qwen3-32b` model and requires `GROQ_API_KEY`.

Add the following to your `.env.local` file:

```env
CONFIDENT_API_KEY="your-confident-api-key"
API_KEY="your-api-key"
```

## Run the Weather Tool-Call Tests

```powershell
.\venv\Scripts\python.exe -m pytest -s -v .\test_whetherToolCall.py
```

## Common Setup and Execution Commands

```powershell
npx skills add confident-ai/confident-skills --skill "confident-onboard"

deepeval login
deepeval view

pip install langchain-community
pip install pytest

# Install and run Ollama locally
pip install langchain-ollama
ollama pull llama3.1
ollama run llama3.1

# Run your agent
python .\agent.py

# Run individual tests
python .\test_firsttest.py
python .\test_secondtest.py
python .\whetherAgent-dynamic.py
python .\test_whetherAgent.py
python .\test_whetherAgent-temperature.py

# Run pytest directly
.\venv\Scripts\python.exe -m pytest -s .\test_whetherAgent-temperature.py
.\venv\Scripts\python.exe -m pytest -s -v .\taskcompleteness.py

# DeepEval tool-calling validation
deepeval test run test_whetherToolCall.py

python .\test_whetherToolCall.py
.\venv\Scripts\python.exe -m pytest -s -v .\test_whetherToolCall.py
```

## Optional: xAI / Grok Remote Inference

```powershell
pip install langchain-xai
```

## Notes

- Use the virtual environment for all test runs.
- Keep API keys in your local environment files and never commit them to version control.
- For historical weather queries, use a different data source or a model that supports time-based weather retrieval.
