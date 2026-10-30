import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from deepeval.test_case import LLMTestCase;
from deepeval import evaluate;
from deepeval.metrics import AnswerRelevancyMetric
from dotenv import load_dotenv, find_dotenv



load_dotenv(find_dotenv())


test_case = LLMTestCase(
    input="What is the capital of France?",
    expected_output="The capital of France is Paris.",
    actual_output="Paris."
)

evaluate([test_case], metrics = [
    AnswerRelevancyMetric()
])