"""
emotion_model.py — Emotion Detection using fine-tuned DistilRoBERTa.

Model: j-hartmann/emotion-english-distilroberta-base
Detects: anger | disgust | fear | joy | neutral | sadness | surprise

Reference: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base
"""

import logging
from utils.config import EMOTION_MODEL_NAME

logger = logging.getLogger(__name__)


# Simple lexicon fallback: word → emotion score boosts
EMOTION_LEXICON = {
    "anger":   ["angry", "furious", "rage", "mad", "hate", "annoyed", "frustrated"],
    "sadness": ["sad", "cry", "depressed", "hopeless", "lonely", "heartbroken", "grief", "miserable"],
    "fear":    ["scared", "afraid", "anxious", "panic", "terrified", "nervous", "worried"],
    "joy":     ["happy", "excited", "great", "awesome", "wonderful", "love", "fantastic", "glad"],
    "disgust": ["disgusting", "gross", "horrible", "awful", "sick", "revolting"],
    "surprise":["wow", "omg", "surprised", "shocked", "unexpected", "unbelievable"],
    "neutral": [],
}


class EmotionDetector:
    """
    Detects emotional tone from user text using a pretrained RoBERTa model.

    Falls back to simple lexicon-based detection if the model is unavailable.
    """

    def __init__(self):
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the Hugging Face emotion classification pipeline."""
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading emotion model: {EMOTION_MODEL_NAME} ...")
            self.pipeline = hf_pipeline(
                task="text-classification",
                model=EMOTION_MODEL_NAME,
                top_k=1,          # return only the top prediction
                device=-1,        # CPU; use device=0 for GPU
                truncation=True,
                max_length=512,
            )
            logger.info("Emotion model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Could not load emotion model: {e}. "
                "Falling back to lexicon-based emotion detection."
            )
            self.pipeline = None

    def _lexicon_fallback(self, text: str) -> str:
        """
        Keyword lexicon-based emotion detection.
        Counts keyword matches per emotion and returns the highest scorer.
        """
        text_lower = text.lower()
        scores = {emotion: 0 for emotion in EMOTION_LEXICON}

        for emotion, keywords in EMOTION_LEXICON.items():
            scores[emotion] = sum(1 for kw in keywords if kw in text_lower)

        best_emotion = max(scores, key=scores.get)

        # If no keywords matched, default to neutral
        if scores[best_emotion] == 0:
            best_emotion = "neutral"

        logger.debug(f"[Lexicon] Emotion: {best_emotion} | Scores: {scores}")
        return best_emotion

    def predict(self, text: str) -> dict:
        """
        Predict the dominant emotion in the user's message.

        Args:
            text: Raw user message string.

        Returns:
            dict with keys:
              - emotion (str): The detected emotion label.
              - score   (float): Confidence score (0–1).
        """
        if not text or not text.strip():
            return {"emotion": "neutral", "score": 1.0}

        # Transformer-based prediction
        if self.pipeline is not None:
            try:
                results = self.pipeline(text)
                # pipeline with top_k=1 returns [[{'label': ..., 'score': ...}]]
                top = results[0][0] if isinstance(results[0], list) else results[0]
                emotion = top["label"].lower()
                score   = round(top["score"], 4)
                logger.debug(f"[RoBERTa] Emotion: {emotion} (score={score})")
                return {"emotion": emotion, "score": score}
            except Exception as e:
                logger.warning(f"Emotion pipeline inference failed: {e}. Using lexicon.")

        # Lexicon fallback
        emotion = self._lexicon_fallback(text)
        return {"emotion": emotion, "score": 0.6}
