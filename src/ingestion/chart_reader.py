"""Chart reader: extract structured numerical evidence from scientific chart images.

Sends a chart image to a Vision-Language Model (VLM), then validates the reply
against a strict Pydantic schema so the result can be stored in the vector
database or passed to the Hypothesis Generator.

Honesty rules enforced in code:

* Every numeric reading carries ``reading_type`` = observed | estimated | uncertain.
  A reading with no value is always ``uncertain``; a value with no explicit type
  defaults to ``estimated`` (never silently ``observed``).
* Non-finite numbers are rejected. Numeric values returned with no supporting
  evidence are flagged in ``uncertainties``.
* Malformed or off-schema model output raises ``InvalidModelResponse``; nothing
  is guessed or repaired.

Configuration (same variables as the Hypothesis agent, with optional VLM_ overrides)::

    VLM_PROVIDER / LLM_PROVIDER   anthropic | openai | gemini     (default: anthropic)
    VLM_API_KEY  / LLM_API_KEY    (gemini also accepts GEMINI_API_KEY / GOOGLE_API_KEY)
    VLM_MODEL    / LLM_MODEL      MUST be a vision-capable model (e.g. a Claude, GPT-4o-mini,
                                  Gemini or LLaVA model). Text-only models fail with an API error.
    VLM_BASE_URL / LLM_BASE_URL   optional; use http://localhost:11434/v1 with provider
                                  "openai" for LLaVA via Ollama (no API key needed on localhost)
    CHART_MAX_IMAGE_MB            optional, default 5

Usage::

    from src.ingestion.chart_reader import extract_chart_information
    result = extract_chart_information("figures/fig3.png")   # -> dict
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ChartReaderError(Exception):
    """Base class for all chart reader errors."""


class ImageValidationError(ChartReaderError):
    """The image path or file is missing, unsupported or corrupt."""


class ChartConfigError(ChartReaderError):
    """Vision model configuration is missing or invalid."""


class ModelAPIError(ChartReaderError):
    """The vision API call failed (network, HTTP error, unexpected response shape)."""


class InvalidModelResponse(ChartReaderError):
    """The model replied, but not with JSON that matches the schema."""


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert scientific chart reader. Analyze the provided scientific chart "
    "image and extract the following information:\n\n"
    "1. Chart type.\n"
    "2. Chart title, if available.\n"
    "3. X-axis and Y-axis labels and units.\n"
    "4. Baseline numerical values.\n"
    "5. Peak performance values.\n"
    "6. Relevant data points and their series labels, when readable.\n"
    "7. Overall trend and a short scientific interpretation.\n"
    "8. Evidence or visual details supporting the extracted values.\n"
    "9. Uncertain or unreadable values.\n\n"
    "Do not invent numerical values. If a value cannot be read confidently, return "
    "null or mark it as uncertain. Distinguish directly observed values from "
    "estimates. Do not claim exact values when the chart does not provide "
    "sufficient information."
)

USER_PROMPT = (
    "Analyze the attached chart image. Reply with ONE JSON object and nothing else, "
    "using exactly these keys:\n"
    '{"chart_title": string|null,\n'
    ' "chart_type": "line"|"bar"|"scatter"|"histogram"|"box"|"heatmap"|"pie"|"area"|"table"|"other",\n'
    ' "x_axis": {"label": string|null, "unit": string|null},\n'
    ' "y_axis": {"label": string|null, "unit": string|null},\n'
    ' "baseline_values": [{"label": string|null, "value": number|null, "unit": string|null, '
    '"series": string|null, "reading_type": "observed"|"estimated"|"uncertain", "note": string|null}],\n'
    ' "peak_performance": same object as a baseline value, or null,\n'
    ' "data_points": [{"series": string|null, "x": number|string|null, "y": number|null, '
    '"reading_type": "observed"|"estimated"|"uncertain", "note": string|null}],\n'
    ' "series": [string],\n'
    ' "trend_summary": string,\n'
    ' "evidence": [string],\n'
    ' "uncertainties": [string]}\n'
    "Rules: use \"observed\" only for values printed on the chart (data labels, tables, "
    "annotations); use \"estimated\" for values read off the axes by eye; use "
    "\"uncertain\" with value null when a number cannot be read. Never invent values. "
    "If the image is not a scientific chart, set chart_type to \"other\" and explain in "
    "uncertainties."
)


# --------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------
ReadingType = Literal["observed", "estimated", "uncertain"]
_CHART_TYPES = ("line", "bar", "scatter", "histogram", "box", "heatmap", "pie", "area", "table", "other")


def _check_finite(value: Optional[float]) -> Optional[float]:
    if value is not None and not math.isfinite(value):
        raise ValueError("numeric values must be finite")
    return value


class AxisInfo(BaseModel):
    label: Optional[str] = None
    unit: Optional[str] = None


class ValueReading(BaseModel):
    """A single numeric reading (baseline or peak)."""

    label: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    series: Optional[str] = None
    reading_type: ReadingType = "estimated"
    note: Optional[str] = None

    @field_validator("value")
    @classmethod
    def _value_finite(cls, v: Optional[float]) -> Optional[float]:
        return _check_finite(v)

    @model_validator(mode="after")
    def _no_value_means_uncertain(self) -> "ValueReading":
        if self.value is None:
            self.reading_type = "uncertain"
        return self


class DataPoint(BaseModel):
    series: Optional[str] = None
    x: Union[float, str, None] = None
    y: Optional[float] = None
    reading_type: ReadingType = "estimated"
    note: Optional[str] = None

    @field_validator("x")
    @classmethod
    def _x_finite(cls, v: Union[float, str, None]) -> Union[float, str, None]:
        if isinstance(v, float):
            _check_finite(v)
        return v

    @field_validator("y")
    @classmethod
    def _y_finite(cls, v: Optional[float]) -> Optional[float]:
        return _check_finite(v)

    @model_validator(mode="after")
    def _no_y_means_uncertain(self) -> "DataPoint":
        if self.y is None:
            self.reading_type = "uncertain"
        return self


class ChartExtraction(BaseModel):
    """Validated result of reading one chart."""

    chart_title: Optional[str] = None
    chart_type: str  # required: an empty/{} reply must not validate
    x_axis: AxisInfo = Field(default_factory=AxisInfo)
    y_axis: AxisInfo = Field(default_factory=AxisInfo)
    baseline_values: List[ValueReading] = Field(default_factory=list)
    peak_performance: Optional[ValueReading] = None
    data_points: List[DataPoint] = Field(default_factory=list)
    series: List[str] = Field(default_factory=list)
    trend_summary: str = ""
    evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)

    @field_validator("chart_type")
    @classmethod
    def _normalise_chart_type(cls, value: str) -> str:
        text = value.strip().lower()
        if not text:
            raise ValueError("must not be empty")
        for known in _CHART_TYPES:
            if text == known or text.startswith(known + " ") or text.endswith(" " + known):
                return known
        if "scatter" in text:
            return "scatter"
        return "other"

    @model_validator(mode="after")
    def _flag_numbers_without_evidence(self) -> "ChartExtraction":
        has_number = (
            any(v.value is not None for v in self.baseline_values)
            or (self.peak_performance is not None and self.peak_performance.value is not None)
            or any(p.y is not None for p in self.data_points)
        )
        if has_number and not self.evidence:
            self.uncertainties.append(
                "Numeric values were returned without supporting visual evidence; treat them as unverified."
            )
        return self


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}


def _sniff_mime(data: bytes) -> Optional[str]:
    """Identify the real image type from its first bytes (the extension can lie)."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def load_image(image_path: Any, max_bytes: int = 5 * 1024 * 1024) -> Tuple[bytes, str]:
    """Validate ``image_path`` and return ``(bytes, mime_type)``."""
    if not isinstance(image_path, (str, os.PathLike)) or not str(image_path).strip():
        raise ImageValidationError("image_path must be a non-empty path.")
    path = Path(image_path).expanduser()
    if not path.exists():
        raise ImageValidationError(f"Image not found: {path}")
    if not path.is_file():
        raise ImageValidationError(f"Not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image format {path.suffix!r}; use one of {sorted(SUPPORTED_EXTENSIONS)}."
        )
    size = path.stat().st_size
    if size == 0:
        raise ImageValidationError(f"Image file is empty: {path}")
    if size > max_bytes:
        raise ImageValidationError(
            f"Image is {size / 1_048_576:.1f} MB; the limit is {max_bytes / 1_048_576:.0f} MB. "
            "Resize or crop the chart."
        )
    data = path.read_bytes()
    mime = _sniff_mime(data)
    if mime is None:
        raise ImageValidationError(f"File is not a valid PNG/JPEG/GIF/WebP image: {path}")
    return data, mime


# --------------------------------------------------------------------------
# Vision client + configuration
# --------------------------------------------------------------------------
class VisionClient(Protocol):
    """Anything with ``describe(system, prompt, image_bytes, mime) -> str`` works."""

    def describe(self, system: str, prompt: str, image_bytes: bytes, mime: str) -> str:  # pragma: no cover
        ...


@dataclass(frozen=True)
class VisionConfig:
    provider: str
    model: str
    api_key: str = field(repr=False)  # never printed or logged
    base_url: Optional[str] = None
    timeout: float = 120.0
    max_tokens: int = 4000


def _first(env: Dict[str, str], *names: str) -> Optional[str]:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def vision_config_from_env(env: Optional[Dict[str, str]] = None) -> VisionConfig:
    """Read VLM settings from the environment (VLM_* first, then the shared LLM_* variables)."""
    if env is None:
        try:  # optional: pick up a local .env file
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
        env = dict(os.environ)
    provider = (_first(env, "VLM_PROVIDER", "LLM_PROVIDER") or "anthropic").lower()
    if provider not in ("anthropic", "openai", "gemini"):
        raise ChartConfigError(
            f"Unsupported provider {provider!r}; set VLM_PROVIDER to anthropic, openai or gemini."
        )
    base_url = _first(env, "VLM_BASE_URL", "LLM_BASE_URL")
    key_names = ["VLM_API_KEY", "LLM_API_KEY"] + (["GEMINI_API_KEY", "GOOGLE_API_KEY"] if provider == "gemini" else [])
    api_key = _first(env, *key_names)
    if not api_key and base_url and (urlparse(base_url).hostname in ("localhost", "127.0.0.1")):
        api_key = "not-needed-for-local-model"  # e.g. LLaVA via Ollama
    missing = []
    if not api_key:
        missing.append(" or ".join(key_names))
    model = _first(env, "VLM_MODEL", "LLM_MODEL")
    if not model:
        missing.append("VLM_MODEL or LLM_MODEL (must be a vision-capable model)")
    if missing:
        raise ChartConfigError("Vision model is not configured. Missing: " + "; ".join(missing) + ".")
    return VisionConfig(provider=provider, model=model, api_key=api_key, base_url=base_url)  # type: ignore[arg-type]


_RETRY_STATUS = {429, 500, 502, 503, 504, 529}


class HTTPVisionClient:
    """Standard-library client for Anthropic, OpenAI-compatible (incl. Ollama) and Gemini vision APIs."""

    def __init__(self, config: VisionConfig, max_retries: int = 2, backoff: float = 2.0) -> None:
        self.config = config
        self.max_retries = max_retries
        self.backoff = backoff

    # -- public ------------------------------------------------------------
    def describe(self, system: str, prompt: str, image_bytes: bytes, mime: str) -> str:
        c = self.config
        b64 = base64.b64encode(image_bytes).decode("ascii")
        if c.provider == "anthropic":
            url = (c.base_url or "https://api.anthropic.com") + "/v1/messages"
            headers = {"content-type": "application/json", "x-api-key": c.api_key,
                       "anthropic-version": "2023-06-01"}
            payload: Dict[str, Any] = {
                "model": c.model, "max_tokens": c.max_tokens, "system": system,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt}]}],
            }
        elif c.provider == "openai":
            url = (c.base_url or "https://api.openai.com/v1") + "/chat/completions"
            headers = {"content-type": "application/json", "authorization": f"Bearer {c.api_key}"}
            payload = {
                "model": c.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
            }
        else:  # gemini
            base = c.base_url or "https://generativelanguage.googleapis.com"
            url = f"{base}/v1beta/models/{c.model}:generateContent"
            headers = {"content-type": "application/json", "x-goog-api-key": c.api_key}
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [
                    {"text": prompt}, {"inline_data": {"mime_type": mime, "data": b64}}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
            }
        data = self._post(url, headers, payload)
        try:
            if c.provider == "anthropic":
                return "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            if c.provider == "openai":
                return data["choices"][0]["message"]["content"] or ""
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelAPIError(
                "Unexpected response shape from the vision API (no text returned; the reply may have "
                f"been blocked or the model may not support images): {type(exc).__name__}"
            ) from exc

    # -- internals ------------------------------------------------------------
    def _redact(self, text: str) -> str:
        return text.replace(self.config.api_key, "***") if self.config.api_key else text

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRY_STATUS and attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                try:
                    detail = exc.read().decode("utf-8", "replace")[:300]
                except Exception:  # pragma: no cover - best effort only
                    detail = ""
                raise ModelAPIError(f"Vision API returned HTTP {exc.code}: {self._redact(detail)}") from exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt < self.max_retries:
                    time.sleep(self.backoff * (2 ** attempt))
                    continue
                raise ModelAPIError(f"Could not reach the vision API: {self._redact(str(exc))}") from exc
            except ValueError as exc:
                raise ModelAPIError("Vision API returned a non-JSON body.") from exc
        raise ModelAPIError("Vision API request failed.")  # pragma: no cover


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def parse_model_response(text: Any) -> ChartExtraction:
    """Parse and validate the model's reply. Raises InvalidModelResponse on any problem."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidModelResponse("The model returned an empty response.")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise InvalidModelResponse("The model response contains no JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as exc:
        raise InvalidModelResponse(f"The model response is not valid JSON: {exc}") from exc
    try:
        return ChartExtraction.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
        raise InvalidModelResponse(f"The model response does not match the chart schema ({problems}).") from exc


def extract_chart_information(
    image_path: str,
    client: Optional[VisionClient] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Read one chart image and return validated, JSON-compatible data.

    Raises a ``ChartReaderError`` subclass (ImageValidationError, ChartConfigError,
    ModelAPIError, InvalidModelResponse) with a clear message on failure.
    ``client`` / ``env`` exist for dependency injection (tests, custom providers).
    """
    max_mb = float((env or os.environ).get("CHART_MAX_IMAGE_MB") or 5)
    image_bytes, mime = load_image(image_path, max_bytes=int(max_mb * 1024 * 1024))
    if client is None:
        client = HTTPVisionClient(vision_config_from_env(env))
    logger.info("Reading chart %s (%s, %d bytes)", Path(image_path).name, mime, len(image_bytes))
    try:
        reply = client.describe(SYSTEM_PROMPT, USER_PROMPT, image_bytes, mime)
    except ChartReaderError:
        raise
    except Exception as exc:  # any custom client failure becomes a clear, typed error
        raise ModelAPIError(f"Vision model call failed: {type(exc).__name__}: {exc}") from exc
    return parse_model_response(reply).model_dump(mode="json")
