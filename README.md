# llm-evaluation
LLM evaluation and agent tool call validation

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the weather tool-call tests

```powershell
.\venv\Scripts\python.exe -m pytest -s -v .\test_whetherToolCall.py
```


## add below to .env 
```
GROQ_API_KEY="your-groq-api-key"
WEATHER_API_KEY="your-openweathermap-api-key"
```

The Groq agent script uses Groq's `qwen/qwen3-32b` model and requires
`GROQ_API_KEY`. Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## add below to .env.local file
```
CONFIDENT_API_KEY
API_KEY
```


The weather tool provides current conditions only; it cannot answer historical
weather questions such as weather over the last seven days.
