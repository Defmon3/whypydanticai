# Structured LLM Output with Pydantic AI

This project demonstrates how to use **[Pydantic AI](https://ai.pydantic.dev/)** to get clean, validated, and schema-guaranteed output from LLMs like **Claude**, **Gemini**, or **OpenAI** — without writing fragile JSON parsing logic.

---

## Why Pydantic AI?

LLMs are great at generating free-form text but notoriously unreliable at producing strict JSON.
Pydantic AI solves this by:

* **Automatic schema enforcement** – every LLM response is parsed and validated against your `pydantic.BaseModel`.
* **Multi-provider support** – swap between Anthropic Claude, Google Gemini, OpenAI GPT, Mistral, etc., by changing *one line*.
* **Typed developer experience** – IntelliSense, type checking, and structured return values.
* **Built-in tool use & agents** – optional; works for single-shot calls or multi-turn workflows.
* **Error transparency** – invalid outputs raise clear `pydantic.ValidationError`s.

---

## Install

```bash

uv add "pydantic-ai-slim[anthropic]" pydantic loguru
```

To swap providers:

```bash

uv add "pydantic-ai-slim[google]"   # for Gemini
uv add "pydantic-ai-slim[openai]"   # for OpenAI
```

---

## Example: Extract & Translate

```python
#!/usr/bin/env python3
"""
SPDX-License-Identifier: LicenseRef-NonCommercial-Only
© 2025 github.com/defmon3 — Non-commercial use only. Commercial use requires permission.

Dependencies:
    uv add "pydantic-ai-slim[anthropic]" pydantic loguru
"""

import os
from pydantic import BaseModel, Field
from loguru import logger as log
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
# from pydantic_ai.models.google import GoogleModel
# from pydantic_ai.models.openai import OpenAIModel


class Extracted(BaseModel):
    full_text: str = Field(min_length=1)
    names: list[str]
    emails: list[str]
    birth_dates: list[str]
    urls: list[str]
    translated_full_text: str
    translated_names: list[str]


def build_agent(target_language: str, provider: str = "anthropic") -> Agent[None, Extracted]:
    instructions = (
        "Extract structured data from user text and produce only valid JSON per the schema. "
        "1) full_text is the input text verbatim. "
        "2) names: all person names found. "
        "3) emails: addresses found. "
        "4) birth_dates: ISO 8601 YYYY-MM-DD if valid. "
        "5) urls: absolute URLs. "
        f"6) translated_full_text: translate to {target_language}. "
        f"7) translated_names: translate to {target_language} if applicable."
    )
    if provider == "anthropic":
        model = AnthropicModel("claude-3.5-sonnet-latest")
    elif provider == "google":
        from pydantic_ai.models.google import GoogleModel
        model = GoogleModel("gemini-1.5-pro")
    elif provider == "openai":
        from pydantic_ai.models.openai import OpenAIModel
        model = OpenAIModel("gpt-4o-mini")
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return Agent(model=model, output_type=Extracted, instructions=instructions)


def extract_and_translate(text: str, target_language: str, provider: str = "anthropic") -> Extracted:
    if not os.getenv("ANTHROPIC_API_KEY") and provider == "anthropic":
        raise RuntimeError("ANTHROPIC_API_KEY env var is required for Anthropic")
    agent = build_agent(target_language, provider)
    res = agent.run_sync(user_prompt=text)
    return res.output


if __name__ == "__main__":
    sample = "Contact John Doe at john@example.com. Born 1990-07-14. Website: https://example.com."
    out = extract_and_translate(sample, "Swedish", provider="anthropic")
    log.info(out.model_dump_json())
```

---

## Switching Providers

Change the `provider` argument in `extract_and_translate()`:

```python
extract_and_translate(sample, "Swedish", provider="google")   # Gemini
extract_and_translate(sample, "Swedish", provider="openai")   # OpenAI
```

All validation and structure remain identical — only the underlying LLM changes.

---

## Environment Variables

Set your API key(s) for the chosen provider:

```bash

$env:ANTHROPIC_API_KEY = "<key>"     # PowerShell example for Claude
$env:GOOGLE_API_KEY = "<key>"        # for Gemini
$env:OPENAI_API_KEY = "<key>"        # for OpenAI
```

---

## Why This Beats Raw API Calls

Without Pydantic AI:

* You write prompt templates to *try* to get JSON back.
* You write brittle parsing + error handling.
* You risk silent failures when the model returns partial/malformed JSON.

With Pydantic AI:

* Model output → `pydantic.BaseModel` in one call.
* Guaranteed type safety.
* Easy provider swapping.
* Cleaner, more maintainable code.
