"""
response_generator.py — LLM Response Generation via Hugging Face Inference API.

Uses the free-tier Hugging Face Inference API (no GPU needed).
Model: google/flan-t5-large (instruction-following, free to use).

Fallback responses are provided if the API is unavailable or returns an error.
"""

import logging
import requests
import time
from utils.config import (
    HF_API_URL,
    HF_API_TOKEN,
    HIGH_RISK_INTENTS,
    HIGH_RISK_EMOTIONS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Curated fallback responses per intent
# These are used when the API is unavailable or rate-limited.
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_RESPONSES = {
    "greeting": (
        "Hey there! I'm MindBridge, your wellness companion. 😊 "
        "I'm here to listen and support you. How are you feeling today?"
    ),
    "academic_stress": (
        "I can hear that academics are weighing heavily on you right now. "
        "That pressure is real and valid. Remember — one step at a time. "
        "Break things into smaller tasks, give yourself credit for what you've done, "
        "and don't hesitate to reach out to your professors or a counselor."
    ),
    "emotional_distress": (
        "I'm really glad you're talking about this. What you're feeling matters. "
        "You don't have to carry this alone. Please consider reaching out to "
        "a campus counselor or someone you trust — you deserve support."
    ),
    "self_harm_risk": (
        "I'm really concerned about you right now, and I want you to know "
        "you matter deeply. Please reach out to a crisis helpline immediately: "
        "iCall: 9152987821 | SNEHI: 044-24640050 | Vandrevala Foundation: 1860-2662-345. "
        "You are not alone, and help is available right now."
    ),
    "general_conversation": (
        "I'm here for you! Feel free to share what's on your mind. "
        "Whether it's stress, emotions, or just a chat — I'm listening. 💙"
    ),
}

DEFAULT_FALLBACK = (
    "I'm here to support you. Whatever you're going through, "
    "you don't have to face it alone. Please feel free to share more. 💙"
)


class ResponseGenerator:
    """
    Generates empathetic responses using the Hugging Face Inference API.

    Flow:
      1. Send the prompt to HF API with generation parameters.
      2. Parse and clean the generated text.
      3. On any failure → return a curated fallback response.
    """

    def __init__(self):
        if not HF_API_TOKEN:
            logger.warning(
                "HF_API_TOKEN not set. API calls will likely fail. "
                "Set env var: export HF_API_TOKEN=hf_xxxxxxxxx"
            )
        self.headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        intent: str,
        emotion: str,
        max_retries: int = 2,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt:      The fully constructed prompt string.
            intent:      Detected intent (used for fallback selection).
            emotion:     Detected emotion.
            max_retries: Number of retry attempts on model loading errors.

        Returns:
            Generated response string.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.75,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.3,
            },
            "options": {
                "wait_for_model": True,  # wait if model is loading (free tier)
                "use_cache": False,
            },
        }

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[ResponseGen] API call attempt {attempt}/{max_retries}")
                response = requests.post(
                    HF_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    return self._parse_response(data, prompt)

                elif response.status_code == 503:
                    # Model is loading on the free tier — wait and retry
                    wait_time = response.json().get("estimated_time", 20)
                    logger.warning(
                        f"[ResponseGen] Model loading (503). "
                        f"Waiting {wait_time:.0f}s before retry..."
                    )
                    time.sleep(min(wait_time, 30))

                elif response.status_code == 401:
                    logger.error("[ResponseGen] Invalid HF API token (401). Check HF_API_TOKEN.")
                    break

                elif response.status_code == 429:
                    logger.warning("[ResponseGen] Rate limited (429). Using fallback.")
                    break

                else:
                    logger.error(
                        f"[ResponseGen] API error {response.status_code}: {response.text[:200]}"
                    )
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"[ResponseGen] Request timed out (attempt {attempt}).")
            except requests.exceptions.ConnectionError:
                logger.error("[ResponseGen] No internet connection. Using fallback.")
                break
            except Exception as e:
                logger.error(f"[ResponseGen] Unexpected error: {e}")
                break

        # All attempts exhausted → use curated fallback
        logger.info(f"[ResponseGen] Using fallback response for intent={intent}")
        return self._get_fallback(intent, emotion)

    def _parse_response(self, data: list | dict, prompt: str) -> str:
        """
        Extract and clean the generated text from the API response.

        Flan-T5 returns: [{"generated_text": "..."}]
        """
        try:
            if isinstance(data, list) and data:
                raw = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                raw = data.get("generated_text", "")
            else:
                raw = str(data)

            # Some models echo the prompt — strip it
            if raw.startswith(prompt):
                raw = raw[len(prompt):].strip()

            # Remove common artifacts
            cleaned = raw.strip().strip('"').strip("'")

            # Ensure minimum meaningful response
            if len(cleaned) < 10:
                logger.warning(f"[ResponseGen] Response too short: {cleaned!r}")
                return DEFAULT_FALLBACK

            logger.info(f"[ResponseGen] Generated response ({len(cleaned)} chars)")
            return cleaned

        except Exception as e:
            logger.error(f"[ResponseGen] Parse error: {e} | data={str(data)[:200]}")
            return DEFAULT_FALLBACK

    def _get_fallback(self, intent: str, emotion: str) -> str:
        """
        Select the most appropriate fallback response.

        High-risk intents/emotions always get the crisis-aware fallback.
        """
        if intent in HIGH_RISK_INTENTS or emotion in HIGH_RISK_EMOTIONS:
            return FALLBACK_RESPONSES.get("self_harm_risk", DEFAULT_FALLBACK)

        return FALLBACK_RESPONSES.get(intent, DEFAULT_FALLBACK)
