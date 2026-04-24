# 🧠 MindBridge — Mental Wellness Chatbot

> A **context-aware AI chatbot for college students** that provides **empathetic mental health support** using NLP and Transformer models.

---

## ✨ Features

MindBridge intelligently detects and responds to:

- 🎯 **Intent** — e.g., stress, greeting, distress
- 💬 **Emotion** — e.g., fear, sadness, joy

It generates **safe, supportive responses** using a Large Language Model with conversation memory.

---

## 📐 Architecture

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
├── app.py                      # Flask API entry point
│
├── chatbot/
│   ├── intent_model.py         # Intent detection (DistilBERT)
│   ├── emotion_model.py        # Emotion detection (RoBERTa)
│   ├── memory.py               # Conversation memory (last N messages)
│   ├── prompt_builder.py       # Prompt construction from intent + emotion
│   └── response_generator.py  # LLM response generation + fallback
│
├── utils/
│   └── config.py               # Configuration and constants
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Tech Stack

| Component | Model / Library |
|---|---|
| Intent Detection | DistilBERT |
| Emotion Detection | RoBERTa |
| Response Generation | Flan-T5 |
| Web Framework | Flask |
| Language | Python |

---


---

## 🧩 Module Overview

| Module | Description |
|---|---|
| `intent_model.py` | Classifies user intent (stress, distress, greeting, etc.) using DistilBERT |
| `emotion_model.py` | Detects emotional tone (fear, sadness, joy, etc.) using RoBERTa |
| `memory.py` | Maintains a rolling window of the last N messages for context |
| `prompt_builder.py` | Constructs a context-aware prompt from intent, emotion, and memory |
| `response_generator.py` | Generates empathetic responses using Flan-T5 with a safe fallback |
| `config.py` | Stores model paths, thresholds, and other configuration values |

---

