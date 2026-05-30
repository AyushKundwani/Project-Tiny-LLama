# 🤖 FinBot — TinyLlama Finance Q&A Chatbot

A beginner-friendly LLM fine-tuning project. Fine-tune TinyLlama on a custom finance dataset using LoRA/QLoRA, serve it with FastAPI, and chat with it in a React UI.

---

## 📁 Project Structure

```
tinyllama-finance-bot/
│
├── dataset/
│   └── finance_qa.jsonl          # 25 finance Q&A training examples
│
├── notebooks/
│   └── finetune.py               # Fine-tuning script (run in Google Colab)
│
├── backend/
│   ├── main.py                   # FastAPI server
│   ├── requirements.txt          # Python dependencies
│   └── finance_adapter/          # ← Put your downloaded LoRA adapter here
│       ├── adapter_config.json
│       └── adapter_model.safetensors
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx              # React entry point
        ├── App.jsx               # Main chat component
        └── App.css               # Styles
```

---

## 🗺️ How Everything Connects

```
[Google Colab]                [Your Machine]
   │                              │
   │  Fine-tune TinyLlama         │
   │  with LoRA on finance data   │
   │                              │
   │──── Download adapter ───────►│
                                  │
                         [FastAPI Backend :8000]
                                  │
                                  │  POST /chat
                                  │  {"question": "What is a stock?"}
                                  │
                         [React Frontend :5173]
                                  │
                          [Your Browser] 🌐
```

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Fine-tune the Model in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change Runtime Type → T4 GPU
4. Upload `dataset/finance_qa.jsonl` to the Colab file system
5. Copy the code from `notebooks/finetune.py` into cells and run each section
6. At the end, download the adapter using:
   ```python
   import shutil
   from google.colab import files
   shutil.make_archive("finance_adapter", "zip", "./finance-tinyllama-lora/final-adapter")
   files.download("finance_adapter.zip")
   ```
7. Unzip `finance_adapter.zip` into `backend/finance_adapter/`

> ⏱️ Expected training time: 5–10 minutes on a free Colab T4 GPU

---

### Step 2: Start the FastAPI Backend

```bash
# Navigate to the backend folder
cd backend

# Create a virtual environment (keeps dependencies isolated)
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate
# Activate it (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

**Expected output:**
```
🖥️  Running on: cuda
⏳ Loading tokenizer...
✅ Tokenizer loaded
⏳ Loading model (first run downloads ~600MB)...
✅ Model loaded
✅ Fine-tuned adapter loaded!
✅ Model ready to answer questions!

INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test the API:**
- Open http://localhost:8000/docs for the interactive Swagger UI
- Try: `GET /health` and `POST /chat`

---

### Step 3: Start the React Frontend

```bash
# Navigate to the frontend folder
cd frontend

# Install Node.js dependencies
npm install

# Start the development server
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in 300ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

Open http://localhost:5173 in your browser. You should see the FinBot chat UI!

---

## 💬 Using the Chatbot

1. The status badge in the top-right should show **"Model Online"** ✅
2. Click any starter question chip, OR type your own question
3. Press **Send** or hit **Enter**
4. The bot will respond in 1–5 seconds (GPU) or 30–60 seconds (CPU)

**Sample questions to try:**
- "What is compound interest?"
- "How does a 401k work?"
- "What is the difference between saving and investing?"
- "Why is diversification important?"
- "What is an index fund?"

---

## 🧠 How Fine-Tuning Works (Simple Explanation)

```
Before fine-tuning:
TinyLlama knows about everything → gives generic answers

After fine-tuning with LoRA:
TinyLlama + Finance Adapter → gives focused finance answers

LoRA trick:
Instead of retraining ALL 1.1 billion weights (expensive!),
we add tiny "adapter" matrices to just the attention layers.
Only ~4 million parameters are trained → fast and cheap!
```

---

## ❓ Common Problems & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `CUDA out of memory` | GPU memory too small | Set `per_device_train_batch_size=1` and `max_seq_length=256` |
| `ModuleNotFoundError: bitsandbytes` | Not installed | Run `pip install bitsandbytes` |
| Backend says "Backend Offline" | API not running | Run `python main.py` in the backend folder |
| Response is garbage text | Wrong prompt format | Make sure prompt uses `### Instruction:` / `### Response:` |
| Training loss not decreasing | Learning rate too low | Try `learning_rate=5e-4` |
| `CORS error` in browser console | CORS not configured | Already handled in `main.py` — check the port numbers match |
| Slow inference on CPU | No GPU | Expected — CPU inference takes 30–60s. Use GPU if possible. |

---

## 📚 Key Concepts Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Fine-tuning** | Teaching a pre-trained model new specific knowledge |
| **LoRA** | A trick to fine-tune only a tiny fraction of model parameters |
| **QLoRA** | LoRA + 4-bit quantization = even less memory usage |
| **Quantization** | Storing model weights with lower precision (4-bit vs 32-bit) to save memory |
| **Adapter** | The small set of trained weights that get added on top of the base model |
| **Tokenizer** | Splits text into small pieces (tokens) the model can process |
| **Epoch** | One complete pass through the entire training dataset |
| **Loss** | How wrong the model's predictions are — lower is better |
| **Inference** | Using the model to generate answers (not training) |

---

## 🔧 Extending This Project

Once you understand the basics, here are ways to level up:

1. **More data**: Add 100+ examples to `finance_qa.jsonl` for better accuracy
2. **Different domain**: Change the dataset to cooking, medicine, law, etc.
3. **Bigger model**: Try `mistralai/Mistral-7B-Instruct-v0.1` (needs more GPU RAM)
4. **Streaming responses**: Use FastAPI's `StreamingResponse` to stream tokens in real-time
5. **Chat history**: Pass previous messages to maintain conversation context
6. **Deploy**: Host the backend on Hugging Face Spaces or Google Cloud Run

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Base Model | TinyLlama 1.1B | Small, fast, runs on free Colab |
| Fine-tuning | LoRA via PEFT | Memory-efficient parameter-efficient tuning |
| Training | TRL SFTTrainer | Designed for instruction fine-tuning |
| Quantization | BitsAndBytes 4-bit | Cuts memory usage by ~75% |
| Backend | FastAPI + Uvicorn | Simple, fast, auto-generates API docs |
| Frontend | React + Vite | Fast development, simple component model |
| Styling | Plain CSS | No framework complexity |
