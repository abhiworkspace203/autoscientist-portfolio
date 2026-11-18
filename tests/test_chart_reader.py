"""Unit tests for the chart reader. No real API key or network is needed:
the vision model is mocked and urlopen is patched. Passing these tests does NOT
prove that a real vision-model call works."""
import base64
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from src.ingestion import chart_reader as cr
from src.ingestion.chart_reader import (
    ChartConfigError, ChartExtraction, ChartReaderError, HTTPVisionClient, ImageValidationError,
    InvalidModelResponse, ModelAPIError, VisionConfig, extract_chart_information, load_image,
    parse_model_response, vision_config_from_env,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
GOOD_REPLY = {
    "chart_title": "Accuracy vs training steps", "chart_type": "line",
    "x_axis": {"label": "Training steps", "unit": None}, "y_axis": {"label": "Accuracy", "unit": "%"},
    "baseline_values": [{"label": "Baseline", "value": 71.2, "unit": "%", "series": "Baseline",
                         "reading_type": "observed", "note": "printed data label"}],
    "peak_performance": {"label": "Ours", "value": 84.0, "unit": "%", "series": "Ours",
                         "reading_type": "estimated", "note": None},
    "data_points": [{"series": "Ours", "x": 1000, "y": 80.5, "reading_type": "estimated"},
                    {"series": "Ours", "x": 2000, "y": None}],
    "series": ["Baseline", "Ours"], "trend_summary": "Accuracy rises then plateaus.",
    "evidence": ["Data label 71.2 next to the Baseline line"], "uncertainties": ["Final point unreadable"],
}
SCHEMA_KEYS = {"chart_title", "chart_type", "x_axis", "y_axis", "baseline_values", "peak_performance",
               "data_points", "series", "trend_summary", "evidence", "uncertainties"}


class FakeVLM:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def describe(self, system, prompt, image_bytes, mime):
        self.calls.append((system, prompt, image_bytes, mime))
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


@pytest.fixture
def png(tmp_path):
    path = tmp_path / "chart.png"
    path.write_bytes(PNG)
    return path


# ---- schema validation -----------------------------------------------------------
def test_valid_extraction_matches_schema_and_is_json_serialisable(png):
    result = extract_chart_information(str(png), client=FakeVLM(GOOD_REPLY))
    assert set(result) == SCHEMA_KEYS
    assert result["chart_type"] == "line" and result["y_axis"] == {"label": "Accuracy", "unit": "%"}
    assert result["peak_performance"]["reading_type"] == "estimated"
    json.dumps(result)  # must not raise


def test_value_none_is_always_uncertain():
    r = ChartExtraction.model_validate(
        {"chart_type": "bar", "peak_performance": {"value": None, "reading_type": "observed"},
         "data_points": [{"y": None, "reading_type": "observed"}]})
    assert r.peak_performance.reading_type == "uncertain" and r.data_points[0].reading_type == "uncertain"


def test_value_without_reading_type_defaults_to_estimated_not_observed():
    r = ChartExtraction.model_validate({"chart_type": "bar", "baseline_values": [{"value": 5}], "evidence": ["x"]})
    assert r.baseline_values[0].reading_type == "estimated"


@pytest.mark.parametrize("raw,expected", [("Line chart", "line"), ("grouped bar", "bar"), ("Scatter plot", "scatter"),
                                           ("BAR", "bar"), ("radar", "other")])
def test_chart_type_normalised(raw, expected):
    assert ChartExtraction(chart_type=raw).chart_type == expected


@pytest.mark.parametrize("payload", [
    {}, {"chart_type": ""}, {"chart_type": "line", "baseline_values": [{"value": "abc"}]},
    {"chart_type": "line", "baseline_values": [{"value": float("inf")}]},
    {"chart_type": "line", "baseline_values": [{"value": 1, "reading_type": "exact"}]},
    {"chart_type": "line", "x_axis": "time"}, {"chart_type": "line", "data_points": [{"y": "high"}]},
    {"chart_type": "line", "evidence": "not a list"}])
def test_invalid_schema_rejected(payload):
    with pytest.raises(Exception):
        ChartExtraction.model_validate(payload)


def test_numbers_without_evidence_are_flagged_as_unverified():
    r = ChartExtraction.model_validate({"chart_type": "line", "baseline_values": [{"value": 3.0}]})
    assert any("without supporting visual evidence" in u for u in r.uncertainties)
    clean = ChartExtraction.model_validate({"chart_type": "line"})
    assert clean.uncertainties == []  # no numbers, nothing to flag


def test_unreadable_chart_reply_is_accepted_as_safe_result(png):
    reply = {"chart_type": "other", "uncertainties": ["Image is a photograph, not a chart"]}
    result = extract_chart_information(str(png), client=FakeVLM(reply))
    assert result["chart_type"] == "other" and result["baseline_values"] == [] and result["peak_performance"] is None


# ---- image path handling ------------------------------------------------------------
@pytest.mark.parametrize("bad", [None, "", "   ", 123, "/no/such/file.png"])
def test_missing_or_invalid_paths(bad):
    with pytest.raises(ImageValidationError):
        extract_chart_information(bad, client=FakeVLM(GOOD_REPLY))


def test_directory_and_unsupported_formats(tmp_path):
    with pytest.raises(ImageValidationError, match="Not a file"):
        load_image(tmp_path)
    for name in ("a.pdf", "a.bmp", "a.txt", "a.svg", "noext"):
        f = tmp_path / name
        f.write_bytes(PNG)
        with pytest.raises(ImageValidationError, match="Unsupported image format"):
            load_image(f)


def test_fake_extension_empty_and_oversized_files(tmp_path):
    fake = tmp_path / "fake.png"
    fake.write_text("this is not an image")
    with pytest.raises(ImageValidationError, match="not a valid"):
        load_image(fake)
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(ImageValidationError, match="empty"):
        load_image(empty)
    big = tmp_path / "big.png"
    big.write_bytes(PNG + b"0" * 2000)
    with pytest.raises(ImageValidationError, match="limit"):
        load_image(big, max_bytes=1000)


def test_mime_comes_from_content_not_extension(tmp_path):
    f = tmp_path / "actually_jpeg.png"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 20)
    assert load_image(f)[1] == "image/jpeg"
    g = tmp_path / "a.webp"
    g.write_bytes(b"RIFF\x00\x00\x00\x00WEBPVP8 ")
    assert load_image(g)[1] == "image/webp"


# ---- configuration ---------------------------------------------------------------------
def test_missing_config_gives_clear_error_naming_variables(png):
    with pytest.raises(ChartConfigError) as err:
        extract_chart_information(str(png), env={})
    assert "VLM_API_KEY" in str(err.value) and "VLM_MODEL" in str(err.value)


def test_missing_model_only():
    with pytest.raises(ChartConfigError, match="VLM_MODEL"):
        vision_config_from_env({"LLM_API_KEY": "k"})


def test_vlm_variables_override_shared_llm_variables():
    cfg = vision_config_from_env({"LLM_API_KEY": "a", "LLM_MODEL": "text-model", "LLM_PROVIDER": "openai",
                                  "VLM_API_KEY": "b", "VLM_MODEL": "vision-model", "VLM_PROVIDER": "gemini"})
    assert (cfg.provider, cfg.model, cfg.api_key) == ("gemini", "vision-model", "b")


def test_shared_llm_variables_work_and_gemini_key_alias():
    assert vision_config_from_env({"LLM_API_KEY": "k", "LLM_MODEL": "m"}).provider == "anthropic"
    assert vision_config_from_env({"VLM_PROVIDER": "gemini", "GEMINI_API_KEY": "g", "VLM_MODEL": "m"}).api_key == "g"


def test_local_ollama_needs_no_key_and_unknown_provider_rejected():
    cfg = vision_config_from_env({"VLM_PROVIDER": "openai", "VLM_BASE_URL": "http://localhost:11434/v1",
                                  "VLM_MODEL": "llava"})
    assert cfg.model == "llava"
    with pytest.raises(ChartConfigError, match="Unsupported provider"):
        vision_config_from_env({"VLM_PROVIDER": "nope", "VLM_API_KEY": "k", "VLM_MODEL": "m"})
    with pytest.raises(ChartConfigError):  # remote host still needs a key
        vision_config_from_env({"VLM_BASE_URL": "https://example.com", "VLM_MODEL": "m"})


def test_api_key_not_in_repr():
    cfg = vision_config_from_env({"LLM_API_KEY": "sk-secret-123", "LLM_MODEL": "m"})
    assert "sk-secret-123" not in repr(cfg)


# ---- invalid model responses ---------------------------------------------------------------
@pytest.mark.parametrize("reply", ["", "   ", None, "I cannot see any chart.", "[1,2]", "{broken", '{"chart_type": 5}',
                                   '{"trend_summary": "no type"}', '{"chart_type":"line","series":"a"}'])
def test_invalid_model_responses(png, reply):
    with pytest.raises(InvalidModelResponse):
        extract_chart_information(str(png), client=FakeVLM(reply))


def test_json_inside_fences_or_prose_is_accepted(png):
    wrapped = "Here is the result:\n```json\n" + json.dumps(GOOD_REPLY) + "\n```\nLet me know!"
    assert extract_chart_information(str(png), client=FakeVLM(wrapped))["chart_type"] == "line"


# ---- API errors and the prompt actually sent -----------------------------------------------------
def test_api_errors_are_typed(png):
    with pytest.raises(ModelAPIError, match="rate limit"):
        extract_chart_information(str(png), client=FakeVLM(error=ModelAPIError("rate limit")))
    with pytest.raises(ModelAPIError, match="RuntimeError"):
        extract_chart_information(str(png), client=FakeVLM(error=RuntimeError("boom")))
    assert issubclass(ModelAPIError, ChartReaderError)


def test_prompt_and_image_are_sent_to_the_model(png):
    vlm = FakeVLM(GOOD_REPLY)
    extract_chart_information(str(png), client=vlm)
    system, prompt, image, mime = vlm.calls[0]
    assert system.startswith("You are an expert scientific chart reader.")
    assert "Do not invent numerical values" in system and "estimates" in system
    assert "reading_type" in prompt and image == PNG and mime == "image/png"


# ---- HTTP client (urlopen patched) ---------------------------------------------------------------------
class Resp:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def patch_urlopen(monkeypatch, responses):
    sent = []

    def fake(request, timeout=None):
        sent.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return Resp(item)

    monkeypatch.setattr(cr.urllib.request, "urlopen", fake)
    monkeypatch.setattr(cr.time, "sleep", lambda s: None)
    return sent


REPLIES = {
    "anthropic": {"content": [{"type": "text", "text": '{"chart_type":"bar"}'}]},
    "openai": {"choices": [{"message": {"content": '{"chart_type":"bar"}'}}]},
    "gemini": {"candidates": [{"content": {"parts": [{"text": '{"chart_type":"bar"}'}]}}]},
}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_http_client_request_shape_per_provider(monkeypatch, provider, png):
    sent = patch_urlopen(monkeypatch, [REPLIES[provider]])
    cfg = VisionConfig(provider=provider, model="vision-x", api_key="SECRET")
    result = extract_chart_information(str(png), client=HTTPVisionClient(cfg))
    assert result["chart_type"] == "bar"
    body = json.loads(sent[0].data)
    b64 = base64.b64encode(PNG).decode()
    assert b64 in json.dumps(body)  # the image really is in the request
    assert "SECRET" not in json.dumps(body) and "SECRET" not in sent[0].full_url  # key only in headers
    assert {"anthropic": "/v1/messages", "openai": "/chat/completions", "gemini": ":generateContent"}[provider] in sent[0].full_url


def test_http_error_is_typed_and_key_redacted(monkeypatch, png):
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key SECRET"}'))
    patch_urlopen(monkeypatch, [err])
    with pytest.raises(ModelAPIError) as exc:
        extract_chart_information(str(png), client=HTTPVisionClient(VisionConfig("openai", "m", "SECRET")))
    assert "401" in str(exc.value) and "SECRET" not in str(exc.value)


def test_retries_on_429_then_succeeds_and_gives_up_after_limit(monkeypatch, png):
    rate = lambda: urllib.error.HTTPError("u", 429, "Too Many", {}, io.BytesIO(b"slow down"))
    patch_urlopen(monkeypatch, [rate(), REPLIES["openai"]])
    ok = extract_chart_information(str(png), client=HTTPVisionClient(VisionConfig("openai", "m", "k")))
    assert ok["chart_type"] == "bar"
    patch_urlopen(monkeypatch, [rate(), rate(), rate()])
    with pytest.raises(ModelAPIError, match="429"):
        extract_chart_information(str(png), client=HTTPVisionClient(VisionConfig("openai", "m", "k")))


def test_network_failure_and_bad_response_shapes(monkeypatch, png):
    client = lambda p: HTTPVisionClient(VisionConfig(p, "m", "k"), max_retries=0)
    patch_urlopen(monkeypatch, [urllib.error.URLError("dns failure")])
    with pytest.raises(ModelAPIError, match="Could not reach"):
        extract_chart_information(str(png), client=client("anthropic"))
    patch_urlopen(monkeypatch, [{"unexpected": True}])
    with pytest.raises(ModelAPIError, match="Unexpected response shape"):
        extract_chart_information(str(png), client=client("gemini"))
    patch_urlopen(monkeypatch, [{"candidates": []}])  # e.g. blocked by safety filters
    with pytest.raises(ModelAPIError):
        extract_chart_information(str(png), client=client("gemini"))


# ---- sample script ---------------------------------------------------------------------------------------
def load_script():
    path = Path(__file__).resolve().parent.parent / "examples" / "run_chart_extraction.py"
    spec = importlib.util.spec_from_file_location("run_chart_extraction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_images(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"fig{i}.png"
        p.write_bytes(PNG + bytes([i]))
        paths.append(str(p))
    return paths


def test_script_refuses_without_config_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "out.json"
    code = load_script().main(["--images", *make_images(tmp_path), "--out", str(out)], env={})
    assert code == 1 and not out.exists() and "Nothing written" in capsys.readouterr().out


def test_script_writes_results_from_model_calls(tmp_path):
    out = tmp_path / "out.json"
    code = load_script().main(["--images", *make_images(tmp_path), "--out", str(out)], client=FakeVLM(GOOD_REPLY))
    data = json.loads(out.read_text())
    assert code == 0 and data["summary"] == {"images_tested": 3, "succeeded": 3, "failed": 0}
    assert all(r["status"] == "ok" and set(r["extraction"]) == SCHEMA_KEYS and len(r["sha256"]) == 64
               for r in data["results"])


def test_script_does_not_write_when_too_few_succeed(tmp_path, capsys):
    out = tmp_path / "out.json"
    images = make_images(tmp_path, 2) + [str(tmp_path / "missing.png")]
    code = load_script().main(["--images", *images, "--out", str(out)], client=FakeVLM(GOOD_REPLY))
    assert code == 2 and not out.exists() and "missing.png: error" in capsys.readouterr().out
