# Voice-Activated AI Business Receptionist (Hybrid RAG)

An intelligent, low-latency automated business receptionist application that implements **Retrieval-Augmented Generation (RAG)**. The application processes voice inquiries through a browser interface, maps semantic queries within an embedded vector store to locate accurate business policies, and leverages a Large Language Model to synthesize highly focused responses.

---

## Technical Architecture

The platform uses a hybrid layout optimized for cost-effectiveness and high scaling:
* **Vector Store Embeddings:** Uses an embedded instance of **ChromaDB** to index corporate knowledge-base records statelessly on startup.
* **Orchestration Layer:** Powered by **FastAPI Framework** handling incoming async operational routing payloads via strict Pydantic parsing filters.
* **LLM Synthesis:** Employs the **Google GenAI SDK** targeting `gemini-2.5-flash` with strict system role isolation instructions to enforce response guardrails.
* **Edge Processing (STT/TTS):** Relies on native Web Speech Recognition APIs directly inside the browser client. This strategy eliminates cloud audio streaming payload latency and avoids cost creep from traditional pay-per-second API voice gateways.

---

## Tech Stack

* **Backend Engine:** FastAPI, Uvicorn, Python
* **Vector Analytics:** ChromaDB Vector Database 
* **Generative AI Platform:** Google Gemini-2.5 Engine
* **Frontend Web APIs:** HTML5 Web Speech Client Layer (SpeechSynthesis / webkitSpeechRecognition)

---

## Installation & Local Operation

1. Clone this repository locally:
   ```bash
   git clone [https://github.com/firered-007-dev/your-repo-name.git](https://github.com/firered-007-dev/your-repo-name.git)
   cd your-repo-name