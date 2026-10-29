"""One place for every call to a language model.

PROVIDER in .env picks the backend (gemini or anthropic). Model names are read
from .env (TRANSCRIBE_MODEL, ANSWER_MODEL), never written in code.

Only the Gemini backend is wired up so far. Anthropic is added if needed.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def require_env(name: str) -> str:
    """Return a setting from .env, or exit with one clear line if it is missing."""
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name} in .env. Copy .env.example to .env and fill it in.")
    return value


def transcribe_image(png_bytes: bytes, prompt: str) -> str:
    """Send one page image and a prompt to the vision model. Return the reply text."""
    provider = os.getenv("PROVIDER", "gemini")
    if provider != "gemini":
        raise SystemExit(f"PROVIDER={provider} is not wired up yet. Use gemini.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=require_env("TRANSCRIBE_MODEL"),
        contents=[
            types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
            prompt,
        ],
        # temperature 0: same page, same answer. Transcription should not be creative.
        config=types.GenerateContentConfig(temperature=0),
    )
    return response.text or ""


def generate_text(prompt: str, max_retries: int = 5) -> dict:
    """Send a text prompt to the answer model. Return {"text", "input_tokens", "output_tokens"}.

    Retries on rate limits and server errors (429, 500, 503) with 1, 2, 4, 8 second waits.
    """
    import time

    provider = os.getenv("PROVIDER", "gemini")
    if provider != "gemini":
        raise SystemExit(f"PROVIDER={provider} is not wired up yet. Use gemini.")

    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    model = require_env("ANSWER_MODEL")
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            usage = response.usage_metadata
            return {
                "text": (response.text or "").strip(),
                "input_tokens": getattr(usage, "prompt_token_count", None),
                "output_tokens": getattr(usage, "candidates_token_count", None),
            }
        except errors.APIError as exc:
            if exc.code not in (429, 500, 503) or attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
