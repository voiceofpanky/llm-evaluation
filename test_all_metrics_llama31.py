import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import requests
from dotenv import find_dotenv, load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

load_dotenv(find_dotenv())

# DeepEval's async metric evaluation can time out against a local Ollama instance.
# Keep the metric runs sequential and give the local model enough time to respond.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE", "60")


def _ollama_available() -> bool:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.skipif(
    not os.getenv("WEATHER_API_KEY"),
    reason="WEATHER_API_KEY is required to call the weather tool.",
)
@pytest.mark.skipif(
    not _ollama_available(),
    reason="A local Ollama server must be running at http://localhost:11434 with llama3.1 available.",
)
def test_weather_metrics_with_llama31():
    agent_path = Path(__file__).with_name("whetherAgent-temperature.py")
    agent_spec = spec_from_file_location("whether_agent_temperature", agent_path)
    if agent_spec is None or agent_spec.loader is None:
        raise ImportError(f"Unable to load weather agent from {agent_path}")

    agent_module = module_from_spec(agent_spec)
    agent_spec.loader.exec_module(agent_module)

    question = "What is the current weather in Tokyo?"
    response = agent_module.agent.invoke({"messages": [{"role": "user", "content": question}]})
    actual_output = response["messages"][-1].content

    weather_result = "Weather in Tokyo: 22.1°C, clear sky, with 67% humidity."
    expected_output = (
        "The current weather in Tokyo is 22.1°C with clear skies and 67% humidity."
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output,
        context=[weather_result],
        retrieval_context=[weather_result],
        tools_called=[
            ToolCall(
                name="get_weather_prediction",
                input_parameters={"location": "Tokyo"},
            )
        ],
        expected_tools=[
            ToolCall(
                name="get_weather_prediction",
                input_parameters={"location": "Tokyo"},
            )
        ],
    )

    llama_model = OllamaModel(model="llama3.1", base_url="http://localhost:11434", temperature=0)

    evaluate(
        test_cases=[test_case],
        metrics=[
            TaskCompletionMetric(
                task="Provide the current weather in Tokyo using the weather tool when needed.",
                model=llama_model,
                threshold=0.7,
                async_mode=False,
            ),
            ToolCorrectnessMetric(
                available_tools=[
                    ToolCall(
                        name="get_weather_prediction",
                        description="Get the current weather for a specified city.",
                        input_parameters={"location": "Tokyo"},
                    )
                ],
                model=llama_model,
                threshold=0.7,
                should_exact_match=True,
                async_mode=False,
            ),
            AnswerRelevancyMetric(model=llama_model, threshold=0.7, async_mode=False),
            GEval(
                name="Output Correctness",
                criteria=(
                    "Determine whether the response correctly reports the current weather for Tokyo, "
                    "including the temperature, weather conditions, and humidity."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.7,
                model=llama_model,
                async_mode=False,
            ),
            FaithfulnessMetric(
                model=llama_model,
                threshold=0.7,
                truths_extraction_limit=5,
                async_mode=False,
            ),
        ],
    )
