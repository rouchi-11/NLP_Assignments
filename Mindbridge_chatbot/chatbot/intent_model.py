"""
intent_model.py — Intent Detection using DistilBERT + Rule-Based Fallback.

Architecture:
  - Primary:  DistilBERT zero-shot classification via Hugging Face pipeline
  - Fallback: Keyword-based rule matching (no model needed)

Detected intents:
  greeting | academic_stress | emotional_distress |
  self_harm_risk | general_conversation
"""

import logging
from typing import Optional
from utils.config import INTENT_MODEL_NAME, INTENT_KEYWORDS

logger = logging.getLogger(__name__)


class IntentDetector:
    """
    Detects the user's intent from their message.

    Strategy:
      1. Try to load DistilBERT zero-shot pipeline (slow first call, cached after).
      2. If model unavailable or fails → fall back to keyword matching.
    """

    CANDIDATE_LABELS = [
        "greeting",
        "academic stress",
        "emotional distress",
        "self harm risk",
        "general conversation",
    ]

    # Map model output labels → clean internal intent keys
    LABEL_MAP = {
        "greeting": "greeting",
        "academic stress": "academic_stress",
        "emotional distress": "emotional_distress",
        "self harm risk": "self_harm_risk",
        "general conversation": "general_conversation",
    }

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Attempt to load DistilBERT zero-shot classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading intent model: {INTENT_MODEL_NAME} ...")
            self.pipeline = hf_pipeline(
                task="zero-shot-classification",
                model=INTENT_MODEL_NAME,
                # Run on CPU; set device=0 for GPU
                device=-1,
            )
            logger.info("Intent model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load DistilBERT pipeline: {e}. "
                "Falling back to keyword-based intent detection."
            )
            self.pipeline = None

    def _keyword_fallback(self, text: str) -> str:
        """
        Rule-based intent detection using keyword matching.
        Iterates intents in priority order; returns first match.
        """
        text_lower = text.lower()

        # Priority order matters — check self-harm first
        priority_order = [
            "self_harm_risk",
            "emotional_distress",
            "academic_stress",
            "greeting",
            "general_conversation",
        ]

        for intent in priority_order:
            keywords = INTENT_KEYWORDS.get(intent, [])
            if any(kw in text_lower for kw in keywords):
                logger.debug(f"[Keyword] Intent matched: {intent}")
                return intent

        # Default fallback
        return "general_conversation"

    def predict(self, text: str) -> str:
        """
        Predict intent from input text.

        Args:
            text: Raw user message string.

        Returns:
            Intent label string (e.g., 'academic_stress').
        """
        if not text or not text.strip():
            return "general_conversation"

        # Try transformer-based classification
        if self.pipeline is not None:
            try:
                result = self.pipeline(
                    text,
                    candidate_labels=self.CANDIDATE_LABELS,
                    multi_label=False,
                )
                # result["labels"][0] is the highest-scoring label
                raw_label = result["labels"][0]
                intent = self.LABEL_MAP.get(raw_label, "general_conversation")
                score = result["scores"][0]
                logger.debug(
                    f"[DistilBERT] Intent: {intent} (score={score:.3f})"
                )
                return intent
            except Exception as e:
                logger.warning(f"Intent pipeline inference failed: {e}. Using keyword fallback.")

        # Keyword fallback
        return self._keyword_fallback(text)
