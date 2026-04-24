```markdown
# 🧠 MindBridge — Mental Wellness Chatbot

A **context-aware AI chatbot for college students** that provides **empathetic mental health support** using NLP and Transformer models.

It detects:
- **Intent** (e.g., stress, greeting, distress)
- **Emotion** (e.g., fear, sadness, joy)  
and generates **safe, supportive responses** using an LLM with conversation memory.

---

## 📐 Architecture (Flow)

```

User Input
↓
Intent Detection (DistilBERT)
↓
Emotion Detection (RoBERTa)
↓
Memory (last N messages)
↓
Prompt Builder
↓
Response Generator (Flan-T5 + fallback)
↓
Chatbot Response

```

---

## 📁 Project Structure

```

chatbot_project/
│
├── app.py                      # Flask API
│
├── chatbot/
│   ├── intent_model.py         # Intent detection
│   ├── emotion_model.py        # Emotion detection
│   ├── memory.py               # Conversation memory
│   ├── prompt_builder.py       # Prompt construction
│   └── response_generator.py   # LLM + fallback
│
├── utils/
│   └── config.py               # Configuration
│
├── requirements.txt
└── README.md

```
```
