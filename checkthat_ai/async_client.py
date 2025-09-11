import os
import logging
from typing import Optional, Union, Mapping

import httpx
from openai import AsyncOpenAI
from openai._exceptions import OpenAIError
from openai._base_client import DEFAULT_MAX_RETRIES
from openai._types import NOT_GIVEN
from httpx import Timeout

from .resources.chat import AsyncChat
from .resources.models import AsyncModels
from ._types import FactCheckResponse, ClaimNormalizationResponse

log: logging.Logger = logging.getLogger(__name__)

class AsyncCheckThatAI(AsyncOpenAI):
    chat: AsyncChat
    models: AsyncModels

    def __init__(
        self,
        api_key: str = None,
        base_url: Optional[str] = None,
        timeout: Union[float, Timeout, None] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs
    ):
        if api_key is None:
            raise OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting one of the following environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, TOGETHER_API_KEY"
            )

        if base_url is None:
            base_url = "https://api.checkthat-ai.com/v1"
            log.debug("base_url not provided, using default: %s", base_url)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            **kwargs,
        )

        self.chat = AsyncChat(self)
        self.models = AsyncModels(self)
        log.info("AsyncCheckThatAI client initialized")

    async def fact_check(self, content: str, **kwargs) -> FactCheckResponse:
        """
        Custom fact-checking method.
        """
        # TODO: Implement actual fact-checking logic
        log.info("Performing fact-check for: %s", content)
        return FactCheckResponse()

    async def normalize_claims(self, text: str, **kwargs) -> ClaimNormalizationResponse:
        """
        Custom claim normalization method.
        """
        # TODO: Implement actual claim normalization logic
        log.info("Normalizing claims for: %s", text)
        return ClaimNormalizationResponse()
