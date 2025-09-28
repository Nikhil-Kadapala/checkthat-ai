from typing import Optional, List, Dict, Any, Union, Literal, TypeVar
from pydantic import BaseModel, Field
from enum import Enum
from openai.types.chat import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion

# Type variable for response format matching OpenAI's pattern
ResponseFormatT = TypeVar("ResponseFormatT")


class EvaluationReport(BaseModel):
    """Evaluation report for post-normalization quality audits."""
    metrics_used: List[str] = Field(description="The evaluation metrics that were applied")
    scores: Dict[str, float] = Field(description="Scores for each metric (0.0 to 1.0 scale)")
    detailed_results: Dict[str, Dict[str, Any]] = Field(description="Detailed results for each metric")
    timestamp: str = Field(description="ISO timestamp when the evaluation was performed")
    report_url: Optional[str] = Field(default=None, description="URL to the full evaluation report if saved to cloud")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="Information about the model used")

class ClaimType(str, Enum):
    ORIGINAL = "original"
    REFINED = "refined"
    FINAL = "final"
class RefinementHistory(BaseModel):
    claim_type: ClaimType = Field(description="The type of claim")
    claim: Optional[str] = Field(default=None, description="The claim")
    score: float = Field(description="Score for the claim (0.0 to 1.0 scale)")
    feedback: Optional[str] = Field(default=None, description="The feedback from the refinement")
class RefinementMetadata(BaseModel):
    """Metadata about the claim refinement process."""
    metric_used: Optional[str] = Field(default=None, description="The metric that was used for refinement")
    threshold: Optional[float] = Field(default=None, description="The threshold that was used for refinement")
    refinement_model: Optional[str] = Field(default=None, description="The model that was used for refinement")
    refinement_history: List[RefinementHistory] = Field(description="History of the refinement process")


class CheckThatChatCompletion(ChatCompletion):
    """Extended ChatCompletion with CheckThat AI evaluation and refinement data."""
    evaluation_report: Optional[EvaluationReport] = Field(
        default=None,
        description="Post-normalization evaluation results when requested"
    )
    refinement_metadata: Optional[RefinementMetadata] = Field(
        default=None,
        description="Metadata about claim refinement process when applied"
    )
    checkthat_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional CheckThat AI-specific metadata"
    )


class CheckThatParsedChatCompletion(ParsedChatCompletion[ResponseFormatT]):
    """Extended ParsedChatCompletion with CheckThat AI evaluation and refinement data."""
    evaluation_report: Optional[EvaluationReport] = Field(
        default=None,
        description="Post-normalization evaluation results when requested"
    )
    refinement_metadata: Optional[RefinementMetadata] = Field(
        default=None,
        description="Metadata about claim refinement process when applied"
    )
    checkthat_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional CheckThat AI-specific metadata"
    )
