from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from whetherAgent import agent


def test_agent_response_to_capital_question():
    question = "how is the whether in japan?"
    response = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    actual_output = response["messages"][-1].content

    test_case = LLMTestCase(
        input=question,
        expected_output=(
            "the current weather in Japan is a clear sky with a temperature of 17.81°C and 82% humidity."           
        ),
        actual_output=actual_output,
    )

    print(f"Agent response: {actual_output}")
    evaluate(
        test_cases=[test_case],
        metrics=[
            GEval(
                name="Geography correctness",
                criteria=(
                    "The response correctly clarifies the weather in Japan, including the temperature and humidity."
                   
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.7,
            )
        ],
    )
