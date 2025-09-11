from typing import Iterable, Union, Optional, Literal, overload
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionChunk
from openai.resources.chat import Completions as OpenAIChatCompletions
from openai.resources.chat import AsyncCompletions as OpenAIAsyncCompletions
from openai._types import NOT_GIVEN
from openai import Stream, AsyncStream

# Assuming ChatModel is not a public type, so using a type alias
ChatModel = str

class Chat:
    def __init__(self, client):
        self.completions = ChatCompletions(client)

class AsyncChat:
    def __init__(self, client):
        self.completions = AsyncChatCompletions(client)

class ChatCompletions(OpenAIChatCompletions):
    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other parameters
        stream: Literal[True],
        **kwargs,
    ) -> Stream[ChatCompletionChunk]:
        ...

    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion:
        ...

    @overload
    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        ...

    def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, Stream[ChatCompletionChunk]]:
        # Add the client's API key to the request body
        if hasattr(self._client, 'api_key') and self._client.api_key:
            # Create a modified kwargs dict that includes the api_key in extra_body
            modified_kwargs = kwargs.copy()
            modified_kwargs['extra_body'] = modified_kwargs.get('extra_body', {})
            if isinstance(modified_kwargs['extra_body'], dict):
                modified_kwargs['extra_body']['api_key'] = self._client.api_key
            kwargs = modified_kwargs

        return super().create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )


class AsyncChatCompletions(OpenAIAsyncCompletions):
    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: Literal[True],
        **kwargs,
    ) -> AsyncStream[ChatCompletionChunk]:
        ...

    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: Literal[False] = False,
        **kwargs,
    ) -> ChatCompletion:
        ...

    @overload
    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        ...

    async def create(
        self,
        *,
        model: Union[str, ChatModel],
        messages: Iterable[ChatCompletionMessageParam],
        # ... other params
        stream: bool = False,
        **kwargs,
    ) -> Union[ChatCompletion, AsyncStream[ChatCompletionChunk]]:
        # Add the client's API key to the request body
        if hasattr(self._client, 'api_key') and self._client.api_key:
            # Create a modified kwargs dict that includes the api_key in extra_body
            modified_kwargs = kwargs.copy()
            modified_kwargs['extra_body'] = modified_kwargs.get('extra_body', {})
            if isinstance(modified_kwargs['extra_body'], dict):
                modified_kwargs['extra_body']['api_key'] = self._client.api_key
            kwargs = modified_kwargs

        return await super().create(
            model=model,
            messages=messages,
            stream=stream,
            **kwargs,
        )
