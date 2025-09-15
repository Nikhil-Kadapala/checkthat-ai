"""
Type definitions for chat completions.
"""

from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel


class ChatCompletionMessageParam(BaseModel):
    """Parameters for a chat completion message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionCreateParams(BaseModel):
    """Parameters for creating a chat completion."""
    messages: List[ChatCompletionMessageParam]
    model: Union[str, Any]  # ChatModel type
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None

    # CheckThat AI specific parameters
    refine_claims: Optional[bool] = None
    post_norm_eval_metrics: Optional[List[str]] = None
    save_eval_report: Optional[bool] = None
    checkthat_api_key: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """A choice in a chat completion response."""
    finish_reason: Optional[str]
    index: int
    message: Dict[str, Any]
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionUsage(BaseModel):
    """Usage information for a chat completion."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletion(BaseModel):
    """A chat completion response."""
    id: str
    choices: List[ChatCompletionChoice]
    created: int
    model: str
    object: str = "chat.completion"
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None
    usage: ChatCompletionUsage
