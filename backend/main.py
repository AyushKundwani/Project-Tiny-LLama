import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Finance Q&A Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paste your Groq API key here
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("✅ Groq model ready!")

class ChatRequest(BaseModel):
    question: str
    max_tokens: int = 500

class ChatResponse(BaseModel):
    question: str
    answer: str
    device: str

@app.get("/")
def root():
    return {"status": "running", "message": "Finance Q&A Chatbot API is live 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "lllama-3.3-70b-versatile"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long.")

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful personal finance assistant. Answer questions clearly and in simple terms."
                },
                {
                    "role": "user",
                    "content": request.question
                }
            ],
            max_tokens=request.max_tokens,
        )
        answer = response.choices[0].message.content.strip()

        return ChatResponse(
            question=request.question,
            answer=answer,
            device="groq-cloud",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)