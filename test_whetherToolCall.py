from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage


_agent_path = Path(__file__).with_name("whetherAgent-temperature.py")
_agent_spec = spec_from_file_location("whether_agent_temperature", _agent_path)
if _agent_spec is None or _agent_spec.loader is None:
    raise ImportError(f"Unable to load weather agent from {_agent_path}")

_agent_module = module_from_spec(_agent_spec)
_agent_spec.loader.exec_module(_agent_module)


@pytest.mark.parametrize(
    ("question", "location"),
    [
        ("What is the current weather in Tokyo?", "Tokyo"),
        ("Will I need a jacket in Paris today?", "Paris"),
    ],
)
def test_weather_question_calls_weather_tool(question, location):
    print(f"\nQuestion: {question}")
    response = _agent_module.agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = response["messages"]
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            print(
                f"Tool called: {tool_call['name']} "
                f"with arguments {tool_call['args']}"
            )

    answer = messages[-1].content
    print(f"Agent answer: {answer}")

    tool_calls = [
        tool_call
        for message in messages
        for tool_call in getattr(message, "tool_calls", [])
    ]

    assert any(
        tool_call["name"] == "get_weather_prediction"
        and location.lower()
        in tool_call["args"]["location"].lower()
        for tool_call in tool_calls
    ), f"Agent did not call the weather tool for {location}."
    assert any(isinstance(message, ToolMessage) for message in messages), (
        "Agent requested the weather tool but no tool result was returned."
    )
