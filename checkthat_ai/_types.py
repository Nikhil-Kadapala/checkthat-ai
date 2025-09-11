from pydantic import BaseModel, Field
from enum import Enum
from typing import Dict, List, Optional, Union

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

__all__ = [
    "FactCheckResponse",
    "ClaimNormalizationResponse",
]