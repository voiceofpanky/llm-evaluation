import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

_agent_path = Path(__file__).with_name("whetherAgent-temperature.py")
_agent_spec = spec_from_file_location("whether_agent_temperature", _agent_path)
if _agent_spec is None or _agent_spec.loader is None:
    raise ImportError(f"Unable to load weather agent from {_agent_path}")

_agent_module = module_from_spec(_agent_spec)
_agent_spec.loader.exec_module(_agent_module)


def test_weather_agent_response_is_correct():
    question = "What is the current weather in Tokyo?"
    actual_output = _agent_module.ask_agent(question)
    print(f"Agent response: {actual_output}")

    evaluate(
        test_cases=[
            LLMTestCase(
                input=question,
                actual_output=actual_output,
            )
        ],
        metrics=[
            GEval(
                name="Weather answer correctness",
                criteria=(
                    "The response answers the question about the current "
                    "weather in Tokyo. It reports weather details returned "
                    "by the weather tool without inventing conditions, or "
                    "clearly says it could not retrieve the weather."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.7,
            )
        ],
    )
