"""
prompt_builder.py — Constructs structured prompts for the LLM.

Responsibilities:
  - Accept detected intent, emotion, context history, and user message.
  - Inject a system persona + safety rules.
  - Build a clean, well-scoped prompt for Flan-T5 / similar instruction LLMs.
  - Apply different prompt templates based on intent (academic / distress / general).
"""

import logging
from utils.config import HIGH_RISK_EMOTIONS, HIGH_RISK_INTENTS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System persona injected at the top of every prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PERSONA = (
    "You are MindBridge, a compassionate and empathetic mental wellness "
    "assistant designed to support college students. You listen carefully, "
    "validate feelings without judgment, and provide thoughtful, supportive "
    "responses. You NEVER provide medical diagnoses or clinical treatment. "
    "If a student appears to be in serious distress, you always encourage "
    "them to speak with a professional counselor or call a crisis helpline."
)

# ─────────────────────────────────────────────────────────────────────────────
# Intent-specific instruction snippets
# ─────────────────────────────────────────────────────────────────────────────
INTENT_INSTRUCTIONS = {
    "greeting": (
        "The student is greeting you. Respond warmly and ask how they are feeling today."
    ),
    "academic_stress": (
        "The student is experiencing academic stress. Acknowledge their pressure, "
        "validate their feelings, and gently offer a practical coping tip or perspective shift. "
        "Remind them it is okay to ask for help."
    ),
    "emotional_distress": (
        "The student appears to be emotionally distressed. Prioritize empathy above all. "
        "Validate their feelings first. Do NOT minimize their experience. "
        "Gently suggest speaking with a campus counselor or a trusted person."
    ),
    "self_harm_risk": (
        "IMPORTANT: The student may be expressing thoughts of self-harm. "
        "Respond with extreme care and compassion. Do NOT be dismissive. "
        "Strongly encourage them to contact a crisis helpline immediately: "
        "iCall (India): 9152987821 | iSTEPP: 080-46110007 | "
        "International: befrienders.org. Stay calm and supportive."
    ),
    "general_conversation": (
        "The student is having a general conversation. Respond helpfully, "
        "warmly, and keep the conversation open and supportive."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Emotion-specific empathy prefix (prepended to the response guidance)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_EMPATHY = {
    "sadness":  "The student seems sad or down. Validate their sadness gently.",
    "fear":     "The student seems anxious or afraid. Reassure them calmly.",
    "anger":    "The student may be frustrated or angry. Acknowledge their frustration without escalating.",
    "joy":      "The student seems positive. Match their energy and be encouraging.",
    "disgust":  "The student seems upset or repulsed by something. Acknowledge their feelings.",
    "surprise": "The student seems surprised. Engage curiously.",
    "neutral":  "Respond in a calm, balanced, and supportive tone.",
}


class PromptBuilder:
    """
    Builds the final LLM prompt from all available signals.

    The prompt structure:
        [System Persona]
        [Safety Rules]
        [Intent Instruction]
        [Emotion Guidance]
        [Conversation Context]
        [Current User Message]
        → Response:
    """

    def build(
        self,
        user_message: str,
        intent: str,
        emotion: str,
        context: str,
        emotion_score: float = 1.0,
    ) -> str:
        """
        Assemble the full prompt string.

        Args:
            user_message:  The raw user input.
            intent:        Detected intent label.
            emotion:       Detected emotion label.
            context:       Formatted conversation history string.
            emotion_score: Confidence of emotion prediction (for logging).

        Returns:
            A formatted prompt string ready to send to the LLM.
        """
        is_high_risk = (
            intent in HIGH_RISK_INTENTS or emotion in HIGH_RISK_EMOTIONS
        )

        # Resolve instruction blocks
        intent_instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["general_conversation"])
        emotion_guidance   = EMOTION_EMPATHY.get(emotion, EMOTION_EMPATHY["neutral"])

        # Build prompt sections
        sections = []

        # ── 1. Persona
        sections.append(f"[PERSONA]\n{SYSTEM_PERSONA}")

        # ── 2. Safety block (always present, emphasized for high risk)
        safety_note = (
            "⚠️  SAFETY RULE: Do NOT diagnose. Do NOT prescribe. "
            "If serious distress is detected, always refer to professional help. "
            "Keep responses under 120 words. Be warm, concise, and human."
        )
        if is_high_risk:
            safety_note = (
                "🚨 HIGH-RISK DETECTED: The student may be in serious distress. "
                "Your ONLY goal is to validate their feelings and connect them "
                "to professional support. Do not offer solutions. Do not minimize. "
                "Keep response short, warm, and action-oriented toward seeking help."
            )
        sections.append(f"[SAFETY]\n{safety_note}")

        # ── 3. Intent context
        sections.append(f"[SITUATION]\nIntent: {intent}\n{intent_instruction}")

        # ── 4. Emotion context
        sections.append(
            f"[EMOTIONAL STATE]\nEmotion: {emotion} (confidence: {emotion_score:.0%})\n"
            f"{emotion_guidance}"
        )

        # ── 5. Conversation history (if any)
        if context.strip():
            sections.append(f"[CONVERSATION HISTORY]\n{context}")

        # ── 6. Current message + response trigger
        sections.append(
            f"[STUDENT SAYS]\n{user_message}\n\n"
            f"[MINDBRIDGE RESPONSE]\nRespond empathetically in 2–4 sentences:"
        )

        prompt = "\n\n".join(sections)

        logger.debug(
            f"[PromptBuilder] Built prompt | intent={intent} emotion={emotion} "
            f"high_risk={is_high_risk} ctx_len={len(context)} chars"
        )
        return prompt

    def build_fallback_prompt(self, user_message: str, intent: str, emotion: str) -> str:
        """
        Minimal prompt for when context is unavailable or as a secondary attempt.
        """
        instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["general_conversation"])
        return (
            f"{SYSTEM_PERSONA}\n\n"
            f"{instruction}\n\n"
            f"Student: {user_message}\n"
            f"MindBridge:"
        )
