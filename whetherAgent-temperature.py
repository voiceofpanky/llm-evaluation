import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

# Load environment variables from .env file
load_dotenv()

# 1. Define the Weather Tool
@tool
def get_weather_prediction(location: str) -> str:
    """Get the current weather prediction and temperature for a given city or location."""
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return f"Weather in {location}: {temp}°C, {desc}, with {humidity}% humidity."
    else:
        return f"Sorry, I couldn't fetch the weather data for {location}. Please check the spelling."

# Package tools into a list
tools = [get_weather_prediction]

# 2. Initialize the Local Open-Source Model via Ollama
model = ChatOllama(model="llama3.1", temperature=0)

# 3. Create the Agent
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful weather assistant. Use the provided weather tool to answer user queries accurately."
)

def ask_agent(question: str) -> str:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return response["messages"][-1].content


if __name__ == "__main__":
    while True:
        try:
            question = input("Ask a weather question (or 'quit' to exit): ").strip()
        except EOFError:
            print()
            break

        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        print(f"Agent: {ask_agent(question)}")