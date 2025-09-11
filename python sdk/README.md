# CheckThat AI Python SDK

A Python SDK for the CheckThat AI fact-checking and claim normalization API.

## Installation

```bash
pip install checkthat-ai
```

## Usage

```python
import os
from checkthat_ai import CheckThatAI

api_key = os.environ.get("OPENAI_API_KEY")

client = CheckThatAI(
    api_key=api_key,
    base_url="https://api.checkthat-ai.com/v1"
)

# Standard OpenAI interface
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Fact-check this claim: The Earth is flat"}
    ]
)

print(response.choices[0].message.content)

print(result)
```

## Example response:

```json
{
    "verdict": "Factually False",
    "evidence": "The Earth is round",
    "sources": ["https://en.wikipedia.org/wiki/Earth"]
}
```