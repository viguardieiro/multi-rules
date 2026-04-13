"""External LLM selector backend support with robust parsing and retries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import time
from typing import Any, Protocol, runtime_checkable
import urllib.error
import urllib.request

from .selector_protocol import InstructionSelector, decision_from_dict
from .types import SelectorDecision, SelectorRequest


DEFAULT_SELECTOR_SYSTEM_PROMPT = (
    "You are an instruction-selector for dynamic attention boosting. "
    "Return ONLY valid JSON with keys: decision, active_instruction_ids, confidence, reason. "
    "decision must be one of stay, switch, add. "
    "active_instruction_ids must be a non-empty subset of candidate_instruction_ids. "
    "confidence must be a float in [0,1]."
)


@runtime_checkable
class SelectorLLMBackend(Protocol):
    """Backend interface used by LLMInstructionSelector."""

    name: str

    def generate(self, system_prompt: str, user_payload: Mapping[str, Any], timeout_s: float) -> str:
        """Return raw model text output for the selector call."""


@dataclass(frozen=True)
class OllamaSelectorBackend:
    """Ollama chat backend implementation for instruction selection."""

    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    extra_options: dict[str, Any] = field(default_factory=dict)
    name: str = "ollama"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model cannot be empty")
        if not self.base_url:
            raise ValueError("base_url cannot be empty")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise TypeError("temperature must be numeric")
        if not isinstance(self.extra_options, dict):
            raise TypeError("extra_options must be a dict")

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/api/chat"

    def generate(self, system_prompt: str, user_payload: Mapping[str, Any], timeout_s: float) -> str:
        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValueError("system_prompt must be a non-empty string")
        if not isinstance(user_payload, Mapping):
            raise TypeError("user_payload must be a mapping")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")

        options = {"temperature": float(self.temperature)}
        options.update(self.extra_options)

        body = {
            "model": self.model,
            "stream": False,
            "options": options,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(dict(user_payload), ensure_ascii=True, indent=2),
                },
            ],
        }

        req = urllib.request.Request(
            self.chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to call Ollama backend: {exc}") from exc

        message = payload.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("Ollama response missing 'message' object")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response missing message content string")
        return content


class SelectorParseError(ValueError):
    """Raised when selector output cannot be parsed to a valid decision."""


@dataclass(frozen=True)
class LLMInstructionSelector(InstructionSelector):
    """InstructionSelector implementation backed by an external LLM backend."""

    backend: SelectorLLMBackend
    system_prompt: str = DEFAULT_SELECTOR_SYSTEM_PROMPT
    timeout_s: float = 30.0
    max_retries: int = 1
    retry_backoff_s: float = 0.0
    max_generation_chars: int = 1600
    raw_output_log_limit_chars: int = 300
    logger: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, SelectorLLMBackend):
            raise TypeError("backend must implement SelectorLLMBackend")
        if not isinstance(self.system_prompt, str) or not self.system_prompt:
            raise ValueError("system_prompt must be a non-empty string")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be a non-negative int")
        if isinstance(self.retry_backoff_s, bool) or not isinstance(self.retry_backoff_s, (int, float)):
            raise TypeError("retry_backoff_s must be numeric")
        if self.retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be >= 0")
        if isinstance(self.max_generation_chars, bool) or not isinstance(self.max_generation_chars, int):
            raise TypeError("max_generation_chars must be an int")
        if self.max_generation_chars < 1:
            raise ValueError("max_generation_chars must be >= 1")
        if isinstance(self.raw_output_log_limit_chars, bool) or not isinstance(self.raw_output_log_limit_chars, int):
            raise TypeError("raw_output_log_limit_chars must be an int")
        if self.raw_output_log_limit_chars < 1:
            raise ValueError("raw_output_log_limit_chars must be >= 1")
        if self.logger is not None and not callable(self.logger):
            raise TypeError("logger must be callable when provided")

    def select(self, request: SelectorRequest) -> SelectorDecision:
        payload = build_selector_payload(request, max_generation_chars=self.max_generation_chars)

        attempts = self.max_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            raw_text = ""
            try:
                raw_text = self.backend.generate(self.system_prompt, payload, timeout_s=float(self.timeout_s))
                decision = parse_selector_response(raw_text)
                decision.validate_candidates(request.candidate_instruction_ids)
                return decision
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.logger is not None:
                    self.logger(
                        f"selector attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}; "
                        f"raw='{sanitize_raw_output(raw_text, self.raw_output_log_limit_chars)}'"
                    )
                if attempt < attempts and self.retry_backoff_s > 0:
                    time.sleep(float(self.retry_backoff_s))

        raise SelectorParseError(f"Selector failed after {attempts} attempts: {last_error}")


@dataclass(frozen=True)
class DeterministicFallbackSelector:
    """Deterministic fallback for selector failures."""

    keep_current_if_possible: bool = True

    def __call__(self, request: SelectorRequest, error: Exception) -> SelectorDecision:
        if self.keep_current_if_possible and request.currently_active_instruction_ids:
            decision = "stay"
            active_ids = request.currently_active_instruction_ids
        else:
            decision = "switch"
            active_ids = [request.candidate_instruction_ids[0]]

        return SelectorDecision(
            decision=decision,
            active_instruction_ids=active_ids,
            confidence=0.0,
            reason=f"fallback_due_to_{type(error).__name__}",
            metadata={"fallback": True, "error_type": type(error).__name__},
        )


def build_selector_payload(request: SelectorRequest, *, max_generation_chars: int = 1600) -> dict[str, Any]:
    """Build compact payload sent to selector backends."""

    if isinstance(max_generation_chars, bool) or not isinstance(max_generation_chars, int) or max_generation_chars < 1:
        raise ValueError("max_generation_chars must be a positive int")

    generation_tail = request.current_generation[-max_generation_chars:]
    return {
        # Give the selector the same input text the model received.
        "model_input": request.base_prompt,
        "step_index": request.step_index,
        "generation_token_count": request.generation_token_count,
        "candidate_instruction_ids": request.candidate_instruction_ids,
        "instruction_texts": [
            {"id": inst_id, "text": request.instruction_text_by_id[inst_id]}
            for inst_id in request.candidate_instruction_ids
        ],
        "currently_active_instruction_ids": request.currently_active_instruction_ids,
        "current_generation": generation_tail,
        "metadata": request.metadata,
    }


def sanitize_raw_output(raw_text: str, max_chars: int = 300) -> str:
    """Sanitize model output for logs without exposing full text blobs."""

    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive int")

    compact = " ".join(raw_text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "..."


def _extract_json_mapping(raw_text: str) -> Mapping[str, Any]:
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")

    stripped = raw_text.strip()
    if not stripped:
        raise SelectorParseError("Empty selector output")

    # Fast path: output is directly valid JSON.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, Mapping):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    required = {"decision", "active_instruction_ids", "confidence"}

    for idx, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped, idx)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, Mapping) and required.issubset(set(parsed.keys())):
            return parsed

    raise SelectorParseError("Could not locate a valid JSON selector object")


def parse_selector_response(raw_text: str) -> SelectorDecision:
    """Parse raw LLM output into a validated SelectorDecision."""

    mapping = _extract_json_mapping(raw_text)
    try:
        return decision_from_dict(mapping)
    except Exception as exc:  # noqa: BLE001
        raise SelectorParseError(f"Parsed JSON is not a valid selector decision: {exc}") from exc
