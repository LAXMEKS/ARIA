ARIA - Adaptive Reasoning Intelligence Assistant


FUll Name : Ai-Powered Voice Assistant with Edge Optimised Model Pipeline


ARIA Phase 2 — what you're building next:

pipeline/
├── wakeword_stage.py    → detect "hey ARIA"
├── stt_stage.py         → audio → text
├── classifier_stage.py  → YOUR INT8 MODEL → intent
├── rag_stage.py         → knowledge_query handler
├── llm_stage.py         → chitchat + device handler
└── tts_stage.py         → speak the response

Each stage handles one job
Intent classifier is the BRAIN that routes everything

User speaks
    ↓
WakeWord detects "hey ARIA"
    ↓
STT converts audio → text
    ↓
Intent Classifier (YOUR INT8 MODEL)
    ↓
Router sends to correct handler:

weather_query  → Weather API → "It's 32°C in Chennai"
device_control → MQTT/GPIO   → turns on light physically
knowledge_query→ RAG + LLM   → answers from your docs
timer_alarm    → threading   → sets actual timer
chitchat       → Ollama LLM  → friendly conversation
    ↓
TTS speaks the response
    ↓
User hears answer

"set a timer for 5 minutes"
  intent: timer_alarm
  entity: duration = 5 minutes  ← extract this!

"turn on the bedroom light"
  intent: device_control
  entity: device = bedroom light  ← extract this!
         action = turn on         ← extract this!

"weather in Chennai tomorrow"
  intent: weather_query
  entity: location = Chennai      ← extract this!
         time     = tomorrow      ← extract this!