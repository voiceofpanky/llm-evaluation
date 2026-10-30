import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

_agent_path = Path(__file__).with_name("whetherAgent-temperature.py")
_agent_spec = spec_from_file_location("whether_agent_temperature", _agent_path)
if _agent_spec is None or _agent_spec.loader is None:
    raise ImportError(f"Unable to load weather agent from {_agent_path}")

_agent_module = module_from_spec(_agent_spec)
_agent_spec.loader.exec_module(_agent_module)


def test_weather_request_completion():
    question = "What is the current weather in Tokyo?"
    response = _agent_module.agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    actual_output = response["messages"][-1].content
    print(f"Agent response: {actual_output}")

    evaluate(
        test_cases=[
            LLMTestCase(
                input=question,
                actual_output=actual_output,
            )
        ],
        metrics=[
            TaskCompletionMetric(
                task=(
                    "Provide the user with current weather information for "
                    "Tokyo, using the weather tool when needed."
                )
            )
        ],
    )
