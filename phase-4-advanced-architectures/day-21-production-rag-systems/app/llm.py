"""
LLM inference module.
Calls a locally running Mistral model via Ollama's REST API.
Ollama must be running: `ollama serve` and model pulled: `ollama pull mistral`
"""

import requests
import json
from app.config import OLLAMA_URL, LLM_MODEL, LLM_TIMEOUT
from app.logger import get_logger

logger = get_logger("llm")


def generate(prompt: str) -> str:
    """
    Send a prompt to the local Ollama Mistral model and return the response.
    Uses streaming=False for simplicity — returns full response at once.
    """
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature for factual, grounded answers
            "top_p": 0.9,
            "num_predict": 512    # max tokens in response
        }
    }

    try:
        logger.info(f"Sending prompt to {LLM_MODEL} ({len(prompt)} chars)")
        response = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        answer = result.get("response", "").strip()
        logger.info(f"LLM response received ({len(answer)} chars)")
        return answer

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running? Run: ollama serve")
        return "Error: LLM service unavailable. Please start Ollama with `ollama serve`."

    except requests.exceptions.Timeout:
        logger.error(f"LLM request timed out after {LLM_TIMEOUT}s")
        return "Error: LLM request timed out."

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return f"Error: {str(e)}"


def health_check() -> bool:
    """Check if Ollama is reachable."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False
