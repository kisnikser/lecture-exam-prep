"""OpenAI-compatible LLM client shared by the extract and synthesize steps.

Talks to vLLM by default and to Anthropic through the same adapter, so the
answer stage never learns which one is behind the endpoint.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from typing import Any, TypeVar

import structlog
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, ValidationError

from examprep.config import LOGS_DIR, get_settings

log = structlog.get_logger()

Model = TypeVar("Model", bound=BaseModel)

# Qwen3.8-Flash-Next thinks by default and wraps the reasoning in <think>.
# The block has to go before anything tries to read the answer as JSON, even
# when thinking is switched off — a model may still emit one.
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMError(RuntimeError):
    """The model did not return something the caller can use."""


def strip_thinking(text: str) -> str:
    without = THINK_RE.sub("", text)
    # An unterminated block means the model ran out of tokens while thinking.
    if "<think>" in without:
        without = without.split("<think>", 1)[0]
    return without.strip()


def extract_json(text: str) -> str:
    """The JSON object inside a reply that may carry prose around it."""

    cleaned = strip_thinking(text)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
    match = JSON_BLOCK_RE.search(cleaned)
    return match.group(0) if match else cleaned


class LLMClient:
    """One client per run, limiting how many requests are in flight."""

    def __init__(self, concurrency: int | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.model = settings.llm_model
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
        self._semaphore = asyncio.Semaphore(concurrency or settings.llm_concurrency)

    async def complete(
        self,
        prompt: str,
        temperature: float,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        options: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not self.settings.llm_thinking:
            # vLLM passes this through to the chat template; harmless elsewhere.
            options["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }

        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(**options)
            except BadRequestError:
                # Not every server supports guided decoding; plain JSON mode
                # still keeps the reply parseable.
                options.pop("response_format", None)
                options.setdefault("extra_body", {})
                response = await self._client.chat.completions.create(**options)

        return response.choices[0].message.content or ""

    async def complete_model(
        self,
        prompt: str,
        schema: type[Model],
        temperature: float,
        max_tokens: int = 4096,
        attempts: int = 2,
    ) -> Model:
        """Ask for JSON and parse it into ``schema``, retrying once on garbage."""

        last_error: Exception | None = None
        raw = ""
        for attempt in range(attempts):
            raw = await self.complete(prompt, temperature, schema=schema, max_tokens=max_tokens)
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning("llm.bad_json", attempt=attempt + 1, error=str(exc)[:200])

        dump_raw(schema.__name__, raw)
        raise LLMError(f"{schema.__name__}: не удалось разобрать ответ модели") from last_error


def dump_raw(label: str, text: str) -> None:
    """Keep the unparseable reply on disk; guessing from a log line is worse."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = LOGS_DIR / f"{label}-{stamp}.txt"
    path.write_text(text, encoding="utf-8")
    log.warning("llm.raw_saved", path=str(path))


async def check_endpoint() -> list[str]:
    """Model names the endpoint reports, so `status` can say it is up."""

    settings = get_settings()
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    models = await client.models.list()
    return [item.id for item in models.data]


def render_prompt(template: str, **values: Any) -> str:
    """Fill ``{placeholders}`` in a prompt file.

    Prompts are Markdown with JSON examples in them, so ``str.format`` is out:
    it would choke on every brace. Placeholders are replaced one by one.
    """

    rendered = template
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        rendered = rendered.replace("{" + key + "}", text)
    return rendered
