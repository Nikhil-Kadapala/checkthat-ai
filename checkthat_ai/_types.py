from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any

class Verdict(str, Enum):
    FACTUALLY_TRUE = "factually true"
    FACTUALLY_FALSE = "factually false"
    PARTIALLY_TRUE = "partially true"
    PARTIALLY_FALSE = "partially false"
    NOT_ENOUGH_INFO = "not enough info"

class FactCheckResponse(BaseModel):
    """
    Represents the response from a fact-checking request.
    """
    verdict: Verdict = Field(description="The result of the fact-checking request.")
    evidence: str = Field(description="The authoritative evidence that supports the verdict.")
    sources: List[str] = Field(description="The sources of the evidence.")

class ClaimNormalizationResponse(BaseModel):
    """
    Represents the response from a claim normalization request.
    """
    claim: List[str] = Field(description="The normalized claim(s) extracted from the input text.")


# Available evaluation metrics for post-normalization quality audits
AVAILABLE_EVAL_METRICS = [
    "G-Eval",
    "Bias",
    "Hallucinations",
    "Hallucination Coverage",
    "Factual Accuracy",
    "Relevance",
    "Coherence"
]

# Type alias for evaluation metrics
EvaluationMetric = Literal[
    "G-Eval",
    "Bias",
    "Hallucinations",
    "Hallucination Coverage",
    "Factual Accuracy",
    "Relevance",
    "Coherence"
]


class EvaluationReport(BaseModel):
    """
    Represents an evaluation report for post-normalization quality audits.
    """
    metrics_used: List[str] = Field(description="The evaluation metrics that were applied")
    scores: Dict[str, float] = Field(description="Scores for each metric (0.0 to 1.0)")
    detailed_results: Dict[str, Dict] = Field(description="Detailed results for each metric")
    timestamp: str = Field(description="ISO timestamp when the evaluation was performed")
    model_info: Dict[str, str] = Field(description="Information about the model used")


# OpenAI-compatible message format
class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ResponseFormat(BaseModel):
    """
    Response format specification for structured outputs.
    """
    type: Literal["json_schema"] = "json_schema"
    json_schema: Optional[Dict[str, Any]] = None


# Note: We use OpenAI's ParsedChatCompletion directly for compatibility
# Request/Response schemas for backend integration are documented in IMPLEMENTATION_SUMMARY.md

def parse_structured_response(response, model_class):
    """
    Helper function to parse structured response from chat completions.

    Args:
        response: ChatCompletion response from client.chat.completions.create()
        model_class: Pydantic model class to parse the response into

    Returns:
        Instance of model_class with parsed data

    Example:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[...],
            response_format=MyModel
        )
        parsed = parse_structured_response(response, MyModel)
    """
    import json
    content = response.choices[0].message.content
    data = json.loads(content)
    return model_class(**data)

__all__ = [
    "FactCheckResponse",
    "ClaimNormalizationResponse",
    "EvaluationReport",
    "EvaluationMetric",
    "AVAILABLE_EVAL_METRICS",
    "OpenAIMessage",
    "ResponseFormat",
    "parse_structured_response",
]