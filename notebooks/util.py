"""Shared vLLM transport, response decoding, and scratch-only JSON storage."""

from copy import deepcopy
import json
from pathlib import Path
from urllib.parse import urlsplit
import urllib.request

OUTPUT_ROOT = Path("/scratch/mingyan_root/mingyan/jmtang/outputs/dynamic_mem/tutorial")


def checked_path(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Run artifacts must stay on scratch under {OUTPUT_ROOT}")
    return resolved


def new_run(path):
    destination = checked_path(path)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def save_json(data, path):
    """Save arbitrary JSON data without overwriting an existing artifact."""
    with checked_path(path).open("x") as handle:
        json.dump(data, handle, indent=2)


def load_json(path):
    """Load JSON without imposing an agent-specific state layout."""
    return json.loads(checked_path(path).read_text())


def decode_text(response):
    """Accept exactly one complete, nonempty text completion."""
    choices = response.get("choices", [])
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Expected one complete, untruncated model response")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned empty content")
    return content


def decode_json(response):
    """Decode JSON without assuming tool names, heartbeat, or memory fields."""
    try:
        return json.loads(decode_text(response))
    except json.JSONDecodeError as error:
        raise ValueError("Model did not return valid JSON") from error


class VLLMClient:
    """Use a local vLLM endpoint with explicit budgets and optional output schema.

    base_url is the server root, without /v1. complete returns text by default;
    response_decoder can instead parse JSON or validate an agent's own protocol.
    schema controls vLLM 0.8.5 guided_json generation, not semantic correctness.
    """

    def __init__(
        self, base_url, output, *, model, max_model_len, max_tokens,
        request_limit, schema=None, response_decoder=decode_text,
        chat_template_kwargs=None, temperature=0.0, seed=42, timeout=120,
    ):
        self.base_url = base_url.rstrip("/")
        address = urlsplit(self.base_url)
        if (
            address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}
            or address.path or address.query or address.fragment or address.username
        ):
            raise ValueError("Use a loopback server root or local SSH tunnel, without /v1")
        for name, value in (
            ("max_model_len", max_model_len), ("max_tokens", max_tokens),
            ("request_limit", request_limit),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_tokens > max_model_len:
            raise ValueError("Response budget exceeds the model context")
        self.output = checked_path(output)
        self.model = model
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.request_limit = request_limit
        self.schema = deepcopy(schema)
        self.response_decoder = response_decoder
        self.chat_template_kwargs = deepcopy(chat_template_kwargs or {})
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.requests = 0

    def request(self, route, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + route, data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def verify_model(self):
        models = self.request("/v1/models")
        if self.model not in [entry["id"] for entry in models["data"]]:
            raise ValueError(f"The service does not expose {self.model}")
        (self.output / "models.json").write_text(json.dumps(models, indent=2))

    def complete(self, messages):
        if self.requests >= self.request_limit:
            raise RuntimeError(f"The {self.request_limit}-request budget is exhausted")
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
            "chat_template_kwargs": self.chat_template_kwargs,
        }
        if self.schema is not None:
            payload["guided_json"] = self.schema
        record = {"index": self.requests, "request": payload}
        try:
            tokenized = self.request("/tokenize", {
                "model": self.model, "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": self.chat_template_kwargs,
            })
            record["prompt_tokens"] = tokenized["count"]
            if tokenized["count"] + self.max_tokens > self.max_model_len:
                raise ValueError("Prompt plus response budget exceeds the server context; no truncation")
            self.requests += 1
            response = self.request("/v1/chat/completions", payload)
            record["response"] = response
            result = self.response_decoder(response)
            record["result"] = result
            return result
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            with (self.output / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
