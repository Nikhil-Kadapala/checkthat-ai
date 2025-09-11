# CheckThat AI Python SDK

[![PyPI version](https://badge.fury.io/py/checkthat-ai.svg)](https://badge.fury.io/py/checkthat-ai)
[![Python Support](https://img.shields.io/pypi/pyversions/checkthat-ai.svg)](https://pypi.org/project/checkthat-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python SDK for the [CheckThat AI](https://checkthat-ai.com) platform's unified LLM API with built-in fact-checking and claim normalization capabilities.

## Features

- 🔄 **Unified LLM Access**: Access 11+ models from OpenAI, Anthropic, Google Gemini, xAI, and Together AI through a single API
- 🔍 **Claim Normalization**: Standardize and structure claims for analysis
- ✅ **Fact-Checking**: Built-in claim verification and evidence sourcing
- 🔌 **OpenAI Compatible**: Drop-in replacement for OpenAI Python SDK
- ⚡ **Async Support**: Full async/await support for high-performance applications
- 🛡️ **Type Safety**: Complete type hints for better development experience

## Installation

```bash
pip install checkthat-ai
```

## Quick Start

### Basic Usage

```python
import os
from checkthat_ai import CheckThatAI

# Initialize the client
api_key = os.environ.get("OPENAI_API_KEY")  # or your provider's API key
client = CheckThatAI(api_key=api_key)

# Use exactly like OpenAI's client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Fact-check this claim: The Earth is flat"}
    ]
)

print(response.choices[0].message.content)
```

### Async Usage

```python
import asyncio
from checkthat_ai import AsyncCheckThatAI

async def main():
    client = AsyncCheckThatAI(api_key="your-api-key")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Streaming Responses

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Supported Models

The SDK provides access to models from multiple providers:

- **OpenAI**: GPT-5, GPT-5 nano, o3, o4-mini
- **Anthropic**: Claude Sonnet 4, Sonnet Opus 4.1
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash
- **xAI**: Grok 4, Grok 3, Grok 3 Mini
- **Together AI**: Llama 3.3 70B, Deepseek R1 Distill Llama 70B

## API Reference

### CheckThatAI Client

```python
client = CheckThatAI(
    api_key="your-api-key",           # Required: Your API key
    base_url="https://api.checkthat-ai.com/v1",  # Optional: Custom base URL
    timeout=30.0,                     # Optional: Request timeout
    max_retries=3,                    # Optional: Max retry attempts
)
```

### Chat Completions

Compatible with OpenAI's chat completions API:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    temperature=0.7,
    max_tokens=1000,
    stream=False,
    # ... other OpenAI parameters
)
```

### Model Information

```python
# List available models
models = client.models.list()
for model in models:
    print(f"Model: {model.id}")
```

***sample response***

```json
{
  "models_list": [
    {
      "provider": "OpenAI",
      "available_models": [
        {
          "name": "GPT-4o",
          "model_id": "gpt-4o-2024-11-20"
        },
        {
          "name": "GPT-4.1",
          "model_id": "gpt-4.1-2025-04-14"
        },
        {
          "name": "o4-mini",
          "model_id": "o4-mini-2025-04-16"
        }
      ]
    },
  ]
}
```

## Authentication

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export XAI_API_KEY="your-xai-key"
export TOGETHER_API_KEY="your-together-key"
```

and pass your API Key to the client:

```python
client = CheckThatAI(api_key="your-api-key")
```

## Error Handling

The SDK uses the same exception types as the OpenAI SDK:

```python
from openai import OpenAIError, RateLimitError, APITimeoutError

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except RateLimitError:
    print("Rate limit exceeded")
except APITimeoutError:
    print("Request timed out")
except OpenAIError as e:
    print(f"API error: {e}")
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/nikhil-kadapala/checkthat-ai/blob/main/CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: kadapalanikhil@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/nikhil-kadapala/checkthat-ai/issues)
- 🌐 Website: [checkthat-ai.com](https://checkthat-ai.com)

## Changelog

See [CHANGELOG.md](https://github.com/nikhil-kadapala/checkthat-ai/releases) for a history of changes.