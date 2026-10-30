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
OPENAI_API_KEY
WEATHER_API_KEY
```

## add below to .env.local file
```
CONFIDENT_API_KEY
API_KEY
```
