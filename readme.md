
python -m venv venv
venv\Scripts\activate
pip install langchain langchain-openai requests python-dotenv
pip install langchain-community


pip install pytest




test command tool calling:
.\venv\Scripts\python.exe -m pytest -s -v .\test_whetherToolCall.py

