"""
config.py — Central configuration for the Mental Wellness Chatbot.
All constants, model names, and API settings live here.
"""

import os

# ─────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────

# Intent detection: lightweight DistilBERT variant
INTENT_MODEL_NAME = "distilbert-base-uncased"

# Emotion detection: fine-tuned RoBERTa on emotions
EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# LLM for response generation (Hugging Face Inference API)
LLM_MODEL_NAME = "google/flan-t5-large"

# ─────────────────────────────────────────────
# Hugging Face Inference API
# ─────────────────────────────────────────────

HF_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_NAME}"

# Set your HF token as env var: export HF_API_TOKEN=hf_xxxxxxxx
# Free-tier tokens work for this project.
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# ─────────────────────────────────────────────
# Conversation Memory
# ─────────────────────────────────────────────

# How many past messages to include as context
CONTEXT_WINDOW = 4

# ─────────────────────────────────────────────
# Flask Settings
# ─────────────────────────────────────────────

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# ─────────────────────────────────────────────
# Intent Labels (rule-based fallback)
# ─────────────────────────────────────────────

INTENT_KEYWORDS = {
    "greeting": [
        "hello", "hi", "hey", "good morning", "good evening",
        "howdy", "sup", "what's up", "greetings"
    ],
    "academic_stress": [
        "exam", "test", "assignment", "deadline", "grades", "study",
        "lecture", "professor", "fail", "failing", "gpa", "semester",
        "homework", "project", "thesis", "dissertation", "college",
        "university", "marks", "result", "pressure", "overwhelmed"
    ],
    "emotional_distress": [
        "sad", "depressed", "lonely", "anxious", "panic", "cry",
        "hopeless", "worthless", "hurt", "pain", "scared", "afraid",
        "terrible", "awful", "miserable", "suffering", "broken",
        "cannot cope", "can't cope", "breakdown", "lost", "empty"
    ],
    "self_harm_risk": [
        "suicide", "kill myself", "end my life", "don't want to live",
        "self harm", "cut myself", "hurt myself", "no reason to live"
    ],
    "general_conversation": [
        "how are you", "what do you do", "tell me", "what is",
        "explain", "help me", "i need", "can you"
    ],
}

# ─────────────────────────────────────────────
# Safety Flags
# ─────────────────────────────────────────────

HIGH_RISK_EMOTIONS = ["sadness", "fear", "disgust"]
HIGH_RISK_INTENTS = ["emotional_distress", "self_harm_risk"]

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

LOG_LEVEL = "DEBUG"
LOG_FORMAT = "[%(asctime)s] %(levelname)s — %(name)s: %(message)s"
