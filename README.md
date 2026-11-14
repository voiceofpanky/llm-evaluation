# LLM-evaluation test with DeepEval and Ollama3.1

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


## execution steps terminal commands 

```
The weather tool provides current conditions only; it cannot answer historical
weather questions such as weather over the last seven days.


npx skills add confident-ai/confident-skills --skill "confident-onboard"                    

deepeval login 
deepeval view
pip install langchain-community 
pip install pytest   

# install ollama pull and run locally 
pip install langchain-ollama   
ollama pull llama3.1 
ollama run llama3.1

# Run your agent
python agent.py  

# trigger first test 
python .\test_firsttest.py  
python .\test_secondtest.py  
python .\whetherAgent-dynamic.py
python .\test_whetherAgent.py

python .\test_whetherAgent-temperature.py.\venv\Scripts\python.exe -m pytest -s .\test_whetherAgent-temperature.py 

 .\venv\Scripts\python.exe -m pytest -s .\test_whetherAgent-temperature.py                                    

 # test tool calling
 python .\test_whetherToolCall.py
           OR
 .\venv\Scripts\python.exe -m pytest -s -v .\test_whetherToolCall.py   

# test task completeness
 .\venv\Scripts\python.exe -m pytest -s -v .\taskcompleteness.py      

```                                            
## Execution steps terminal commands (optional for Xai /grok models for remote inference.)
                                       pip install langchain-xai