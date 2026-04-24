"""
app.py — Flask REST API for the Mental Wellness Chatbot.

Endpoints:
  POST /chat          — Main chat endpoint
  GET  /health        — Health check
  GET  /history/<id>  — Retrieve session history
  DELETE /session/<id> — Clear a session

All heavy models are loaded once at startup (lazy via class __init__).
"""

import logging
import uuid
from flask import Flask, request, jsonify

# ── Internal modules
from chatbot.intent_model      import IntentDetector
from chatbot.emotion_model     import EmotionDetector
from chatbot.memory            import MemoryStore
from chatbot.prompt_builder    import PromptBuilder
from chatbot.response_generator import ResponseGenerator
from utils.config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    CONTEXT_WINDOW, LOG_FORMAT, LOG_LEVEL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL, "DEBUG"), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Global Components (loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────
logger.info("Initializing chatbot components...")

intent_detector    = IntentDetector()
emotion_detector   = EmotionDetector()
memory_store       = MemoryStore(max_turns_per_session=20)
prompt_builder     = PromptBuilder()
response_generator = ResponseGenerator()

logger.info("All components ready. MindBridge is online. 💙")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Quick health-check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "MindBridge Mental Wellness Chatbot",
        "version": "1.0.0",
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Request JSON:
        {
            "message":    "I'm really stressed about my exams",  # required
            "session_id": "abc-123"                              # optional
        }

    Response JSON:
        {
            "session_id": "abc-123",
            "intent":     "academic_stress",
            "emotion":    "fear",
            "response":   "...",
            "is_high_risk": false
        }
    """
    # ── 1. Parse input
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Field 'message' is required and cannot be empty."}), 400

    # ── 2. Session management
    session_id = data.get("session_id") or str(uuid.uuid4())
    memory     = memory_store.get_or_create(session_id)

    logger.info(f"[/chat] session={session_id} | msg={user_message[:80]!r}")

    # ── 3. Intent Detection
    intent = intent_detector.predict(user_message)
    logger.info(f"[/chat] Intent: {intent}")

    # ── 4. Emotion Detection
    emotion_result = emotion_detector.predict(user_message)
    emotion        = emotion_result["emotion"]
    emotion_score  = emotion_result["score"]
    logger.info(f"[/chat] Emotion: {emotion} ({emotion_score:.0%})")

    # ── 5. Retrieve Conversation Context
    context = memory.get_context(last_n=CONTEXT_WINDOW)

    # ── 6. Build LLM Prompt
    prompt = prompt_builder.build(
        user_message  = user_message,
        intent        = intent,
        emotion       = emotion,
        context       = context,
        emotion_score = emotion_score,
    )

    # ── 7. Generate Response
    bot_response = response_generator.generate(
        prompt  = prompt,
        intent  = intent,
        emotion = emotion,
    )

    # ── 8. Update Memory
    memory.add_user_message(user_message)
    memory.add_assistant_message(bot_response)

    # ── 9. Determine risk flag
    from utils.config import HIGH_RISK_INTENTS, HIGH_RISK_EMOTIONS
    is_high_risk = intent in HIGH_RISK_INTENTS or emotion in HIGH_RISK_EMOTIONS

    # ── 10. Build and return response
    response_payload = {
        "session_id":   session_id,
        "intent":       intent,
        "emotion":      emotion,
        "emotion_score": emotion_score,
        "response":     bot_response,
        "is_high_risk": is_high_risk,
    }

    if is_high_risk:
        response_payload["safety_resources"] = {
            "iCall (India)":              "9152987821",
            "Vandrevala Foundation":      "1860-2662-345",
            "SNEHI":                      "044-24640050",
            "International (Befrienders)":"befrienders.org",
        }

    logger.info(f"[/chat] Response sent | risk={is_high_risk}")
    return jsonify(response_payload), 200


@app.route("/history/<session_id>", methods=["GET"])
def get_history(session_id: str):
    """
    Retrieve full conversation history for a session.

    Response JSON:
        {
            "session_id": "...",
            "turn_count": 4,
            "history": [...]
        }
    """
    memory = memory_store.get_or_create(session_id)
    return jsonify({
        "session_id": session_id,
        "turn_count": len(memory),
        "history":    memory.get_all_turns(),
    }), 200


@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Clear and remove a session."""
    deleted = memory_store.delete_session(session_id)
    if deleted:
        return jsonify({"message": f"Session {session_id!r} deleted."}), 200
    return jsonify({"error": f"Session {session_id!r} not found."}), 404


@app.route("/sessions", methods=["GET"])
def list_sessions():
    """List all active session IDs (for debugging)."""
    return jsonify({"active_sessions": memory_store.active_sessions()}), 200


# ─────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error. Please try again."}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host  = FLASK_HOST,
        port  = FLASK_PORT,
        debug = FLASK_DEBUG,
    )
