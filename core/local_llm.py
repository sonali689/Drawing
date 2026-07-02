"""
Local LLM Integration via Ollama
==================================
Drop-in replacement for the Anthropic API calls, running entirely locally.

Uses Ollama (https://ollama.com) to serve local vision and text models.
Recommended models:
  - LLaVA 1.6 (7B or 13B) for vision tasks (reading/analyzing drawing images)
  - Mistral / Llama 3 for text-only reasoning (when OCR has already extracted text)

Same prompt interface as the Anthropic code in change_verifier.py etc,
so both backends can be swapped via a single toggle.

Ollama must be installed and running locally: https://ollama.com/download
After install, pull a model:
    ollama pull llava:7b
    ollama pull llava:13b
"""
import base64
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

# Default models — user can override via sidebar or environment
DEFAULT_VISION_MODEL = "llava:7b"
DEFAULT_TEXT_MODEL = "llama3:8b"

_ollama_available = None


def check_ollama() -> bool:
    """Check if Ollama is installed and the server is reachable."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    try:
        import ollama as _ollama
        # Try listing models — this will fail if Ollama server isn't running
        _ollama.list()
        _ollama_available = True
    except Exception:
        _ollama_available = False
    return _ollama_available


def list_models() -> List[str]:
    """List available Ollama models."""
    if not check_ollama():
        return []
    try:
        import ollama
        response = ollama.list()
        return [m.model for m in response.models]
    except Exception:
        return []


def has_vision_model() -> bool:
    """Check if any vision-capable model is available."""
    models = list_models()
    vision_models = {"llava", "llava:7b", "llava:13b", "llava:34b",
                     "bakllava", "moondream", "llava-phi3"}
    return any(m.split(":")[0] in vision_models or m in vision_models for m in models)


def _encode_image(img: np.ndarray) -> str:
    """Encode a BGR image to base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode image as PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Vision call (image + text prompt → text response)
# ---------------------------------------------------------------------------

def vision_call(images: List[np.ndarray], prompt: str,
                model: str = None, temperature: float = 0.1,
                max_tokens: int = 2000) -> str:
    """
    Send images + text prompt to a local vision model via Ollama.

    Args:
        images: List of BGR numpy arrays.
        prompt: Text prompt.
        model: Ollama model name (default: llava:7b).
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's text response.
    """
    import ollama

    model = model or os.environ.get("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL)

    # Encode images to base64
    image_b64_list = [_encode_image(img) for img in images]

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": image_b64_list,
        }],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    return response["message"]["content"]


def text_call(prompt: str, model: str = None,
              temperature: float = 0.1,
              max_tokens: int = 2000) -> str:
    """
    Send a text-only prompt to a local LLM via Ollama.

    Args:
        prompt: Text prompt.
        model: Ollama model name (default: llama3:8b).
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        The model's text response.
    """
    import ollama

    model = model or os.environ.get("OLLAMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)

    response = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def parse_json_response(text: str) -> dict:
    """
    Extract JSON from an LLM response that might contain markdown fences
    or surrounding text.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        # Find content between ``` markers
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Try to find JSON object or array in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


# ---------------------------------------------------------------------------
# Convenience wrappers matching the Anthropic call signatures
# ---------------------------------------------------------------------------

def vision_call_json(images: List[np.ndarray], prompt: str,
                     model: str = None, **kwargs) -> dict:
    """Vision call that parses the response as JSON."""
    raw = vision_call(images, prompt, model=model, **kwargs)
    return parse_json_response(raw)


def text_call_json(prompt: str, model: str = None, **kwargs) -> dict:
    """Text call that parses the response as JSON."""
    raw = text_call(prompt, model=model, **kwargs)
    return parse_json_response(raw)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------

class AIBackend:
    """
    Unified interface for AI calls — supports both Ollama (local) and
    Anthropic (cloud). Modules import this class and call its methods
    without worrying about which backend is active.
    """

    def __init__(self, backend: str = "ollama",
                 api_key: Optional[str] = None,
                 vision_model: Optional[str] = None,
                 text_model: Optional[str] = None):
        """
        Args:
            backend: "ollama" for local, "anthropic" for cloud.
            api_key: Required for "anthropic" backend.
            vision_model: Override default vision model name.
            text_model: Override default text model name.
        """
        self.backend = backend
        self.api_key = api_key
        self.vision_model = vision_model
        self.text_model = text_model
        self._anthropic_client = None

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            )
        return self._anthropic_client

    def call_vision(self, images: List[np.ndarray], prompt: str,
                    max_tokens: int = 2000) -> str:
        """Send images + prompt, get text response."""
        if self.backend == "ollama":
            return vision_call(images, prompt,
                               model=self.vision_model,
                               max_tokens=max_tokens)
        else:
            return self._anthropic_vision(images, prompt, max_tokens)

    def call_vision_json(self, images: List[np.ndarray], prompt: str,
                         max_tokens: int = 2000) -> dict:
        """Send images + prompt, get parsed JSON response."""
        raw = self.call_vision(images, prompt, max_tokens=max_tokens)
        return parse_json_response(raw)

    def call_text(self, prompt: str, max_tokens: int = 2000) -> str:
        """Send text prompt, get text response."""
        if self.backend == "ollama":
            return text_call(prompt, model=self.text_model,
                             max_tokens=max_tokens)
        else:
            return self._anthropic_text(prompt, max_tokens)

    def call_text_json(self, prompt: str, max_tokens: int = 2000) -> dict:
        """Send text prompt, get parsed JSON response."""
        raw = self.call_text(prompt, max_tokens=max_tokens)
        return parse_json_response(raw)

    def _anthropic_vision(self, images: List[np.ndarray], prompt: str,
                          max_tokens: int) -> str:
        client = self._get_anthropic_client()
        model = self.vision_model or "claude-sonnet-4-20250514"

        content = []
        for i, img in enumerate(images):
            b64 = _encode_image(img)
            if i > 0:
                content.append({"type": "text", "text": f"Image {i + 1}:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": b64},
            })
        content.append({"type": "text", "text": prompt})

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(
            block.text for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()

    def _anthropic_text(self, prompt: str, max_tokens: int) -> str:
        client = self._get_anthropic_client()
        model = self.text_model or "claude-sonnet-4-20250514"

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()
