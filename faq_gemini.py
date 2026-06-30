import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types

load_dotenv()

app = FastAPI(title="Voice-Ready FAQ Receptionist")

# Verify API Keys
if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

# Initialize clients (ChromaDB runs statelessly in-memory here)
gemini_client = genai.Client()
chroma_client = chromadb.Client()
faq_collection = chroma_client.get_or_create_collection(name="business_faq")

# Seed ChromaDB on startup
@app.on_event("startup")
def initialize_knowledge_base():
    # Only seed if collection is empty
    if faq_collection.count() == 0:
        faq_data = [
            {"id": "faq1", "text": "Our business hours are Monday through Friday from 9:00 AM to 6:00 PM. We are closed on weekends."},
            {"id": "faq2", "text": "The company headquarters is located at 500 Innovation Way, Suite 100, San Francisco, California."},
            {"id": "faq3", "text": "For support requests, customers can reach us via email at support@techcorp.com or call 1-800-555-0199."},
            {"id": "faq4", "text": "We offer a 30-day money-back refund policy on all software licenses and hardware purchases."},
            {"id": "faq5", "text": "Our current CEO is Dr. Aris Vance, and the company was founded in the year 2021."}
        ]
        for item in faq_data:
            faq_collection.add(documents=[item["text"]], ids=[item["id"]])

class ChatRequest(BaseModel):
    text: str

# Web API Endpoint for the AI Agent
@app.post("/api/ask")
async def ask_receptionist(payload: ChatRequest):
    user_text = payload.text.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
        
    # 1. Vector Search Context matching
    results = faq_collection.query(query_texts=[user_text], n_results=1)
    matched_context = "No matching documentation found."
    if results and 'documents' in results and len(results['documents'][0]) > 0:
        matched_context = results['documents'][0][0]
        
    # 2. Synthesize response using Gemini
    system_instruction = (
        "You are an elegant, professional front-desk AI receptionist. "
        "Answer the question using ONLY the provided business context block. "
        "Keep your answer completely concise (1-2 sentences maximum) and natural. "
        "If the context doesn't answer it, say you don't have that info and offer human support."
        f"\n\nBUSINESS CONTEXT:\n{matched_context}"
    )
    
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_text,
            config=types.GenerateContentConfig(system_instruction=system_instruction)
        )
        ai_text = response.text.strip() if response.text else "I am sorry, could you repeat that?"
        return {"response": ai_text, "context_used": matched_context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Frontend Interface with Web Speech TTS/STT built in
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Receptionist</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; background: #0f172a; color: #f8fafc; padding-top: 60px; }
        .container { max-width: 600px; margin: 0 auto; background: #1e293b; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }
        button { padding: 12px 24px; font-size: 16px; cursor: pointer; border-radius: 8px; border: none; background: #3b82f6; color: white; font-weight: bold; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        #status { margin-top: 20px; color: #94a3b8; font-style: italic; }
        #output { margin-top: 25px; font-size: 18px; color: #38bdf8; font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice AI Business Receptionist</h1>
        <p>Click the button below and ask a business question (e.g., "What are your hours?")</p>
        <button id="talkBtn">Push to Talk</button>
        <p id="status">Ready.</p>
        <div id="output"></div>
    </div>

    <script>
        const talkBtn = document.getElementById('talkBtn');
        const status = document.getElementById('status');
        const output = document.getElementById('output');

        // Browser Native Speech-to-Text Setup
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            status.textContent = "Your browser does not support Web Speech Recognition.";
            talkBtn.disabled = true;
        } else {
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';

            talkBtn.onclick = () => {
                recognition.start();
                status.textContent = "Listening... Speak into your mic.";
                output.textContent = "";
            };

            recognition.onresult = async (event) => {
                const speechToText = event.resultNoMatch ? "" : event.results[0][0].transcript;
                status.textContent = `Processing question: "${speechToText}"...`;
                
                // Send transcript text to Railway FastAPI backend
                const res = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: speechToText })
                });
                const data = await res.json();
                
                if(data.response) {
                    output.textContent = "🤖 Jarvis: " + data.response;
                    status.title = "Context used: " + data.context_used;
                    status.textContent = "Speaking answer...";
                    
                    // Browser Native Text-to-Speech playback
                    const utterance = new SpeechSynthesisUtterance(data.response);
                    window.speechSynthesis.speak(utterance);
                    utterance.onend = () => { status.textContent = "Ready."; };
                } else {
                    status.textContent = "Error processing response.";
                }
            };
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def root():
    return HTMLResponse(HTML_CONTENT)