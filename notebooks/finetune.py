# ============================================================
# TinyLlama Finance Q&A Fine-Tuning Notebook
# Run this in Google Colab (GPU runtime recommended)
# ============================================================

# ─────────────────────────────────────────────
# STEP 0: Install required libraries
# ─────────────────────────────────────────────
# What: Installs all the Python packages we need.
# Why:  Google Colab doesn't have these pre-installed.
#
# Run this in a Colab cell:
# !pip install -q transformers datasets peft trl bitsandbytes accelerate

# ─────────────────────────────────────────────
# STEP 1: Import libraries
# ─────────────────────────────────────────────
# What: Loads all tools into memory.
# Why:  We need these specific libraries to load models,
#       apply LoRA, and run the training loop.

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,          # Converts text → numbers the model understands
    AutoModelForCausalLM,   # Loads the TinyLlama language model
    TrainingArguments,      # Controls how training runs (epochs, batch size, etc.)
    BitsAndBytesConfig,     # Enables 4-bit quantization to save GPU memory
)
from peft import (
    LoraConfig,             # Defines which parts of the model LoRA will tune
    get_peft_model,         # Wraps the model with LoRA adapters
    TaskType,               # Tells PEFT we're doing text generation
)
from trl import SFTTrainer  # A trainer designed specifically for instruction fine-tuning


# ─────────────────────────────────────────────
# STEP 2: Check GPU availability
# ─────────────────────────────────────────────
# What: Checks if a GPU is available. If not, uses CPU.
# Why:  Fine-tuning is very slow on CPU. Colab's free GPU (T4) works great.
# Common mistake: Forgetting to enable GPU runtime in Colab.
#   → Go to Runtime > Change Runtime Type > GPU

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")
# Expected output: ✅ Using device: cuda


# ─────────────────────────────────────────────
# STEP 3: Load the dataset
# ─────────────────────────────────────────────
# What: Reads our custom finance Q&A JSONL file and loads it.
# Why:  The model needs examples in the right format to learn from.
# 
# Our dataset format (each line in finance_qa.jsonl):
# {"instruction": "What is a stock?", "input": "", "output": "A stock is..."}

def load_jsonl(filepath):
    """Reads a .jsonl file and returns a list of dictionaries."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                data.append(json.loads(line))
    return data

# Load our finance dataset
raw_data = load_jsonl("finance_qa.jsonl")
print(f"✅ Loaded {len(raw_data)} examples")
# Expected output: ✅ Loaded 25 examples


# ─────────────────────────────────────────────
# STEP 4: Format data into instruction prompts
# ─────────────────────────────────────────────
# What: Converts each Q&A pair into a structured text prompt.
# Why:  TinyLlama was pre-trained to follow this specific "Alpaca" prompt
#       format. Using the same format helps it understand our instructions.
#
# The Alpaca format looks like:
# ### Instruction:
# What is a stock?
# ### Response:
# A stock is...

def format_prompt(example):
    """Converts a Q&A dict into the Alpaca instruction format."""
    instruction = example["instruction"]
    input_text  = example.get("input", "")     # Optional extra context
    output      = example["output"]

    # Build the prompt text
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    return {"text": prompt}  # SFTTrainer expects a "text" column

# Apply formatting to all examples
formatted_data = [format_prompt(ex) for ex in raw_data]

# Convert list to a Hugging Face Dataset object
dataset = Dataset.from_list(formatted_data)
print(f"✅ Dataset ready. Sample:\n{dataset[0]['text'][:200]}")


# ─────────────────────────────────────────────
# STEP 5: Configure 4-bit quantization (QLoRA)
# ─────────────────────────────────────────────
# What: Sets up 4-bit loading so the model uses much less GPU memory.
# Why:  TinyLlama in full precision needs ~4GB VRAM. With 4-bit, it needs ~1GB.
#       Colab free tier only gives ~15GB VRAM, so this is very important.
#
# Common mistake: Using bfloat16 on older GPUs that don't support it.
#   → We use float16 as it works on all Colab GPUs (T4, V100, A100).

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # Load model weights in 4-bit
    bnb_4bit_quant_type="nf4",                  # NF4 is the best quantization type
    bnb_4bit_compute_dtype=torch.float16,       # Use float16 for calculations
    bnb_4bit_use_double_quant=True,             # Extra memory saving (nested quant)
)
print("✅ Quantization config ready")


# ─────────────────────────────────────────────
# STEP 6: Load the tokenizer and model
# ─────────────────────────────────────────────
# What: Downloads TinyLlama and its tokenizer from Hugging Face Hub.
# Why:  The tokenizer splits text into tokens. The model generates responses.
#
# TinyLlama is only 1.1 billion parameters — small enough for free Colab!
# Full-size Llama 2 has 7B–70B parameters, which would be too large.
#
# Expected download: ~600MB first time (cached after that)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print(f"⏳ Loading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# TinyLlama doesn't have a pad token by default — we need to add one.
# Without this you'll get: "ValueError: Asking to pad but the tokenizer does not have a padding token"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Pad on the right side for training
print("✅ Tokenizer loaded")

print(f"⏳ Loading model from {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,    # Apply 4-bit quantization
    device_map="auto",                  # Automatically put layers on GPU/CPU
    trust_remote_code=True,             # Allow model's custom code to run
)
model.config.use_cache = False          # Disable KV cache during training (saves memory)
print("✅ Model loaded")
print(f"   Model parameters: {model.num_parameters():,}")
# Expected: ~1.1 billion parameters


# ─────────────────────────────────────────────
# STEP 7: Configure LoRA adapters
# ─────────────────────────────────────────────
# What: Defines which parts of the model LoRA will add trainable weights to.
# Why:  Instead of training all 1.1B parameters (slow & expensive),
#       LoRA only trains ~0.5% of parameters by adding small adapter matrices.
#       This makes fine-tuning fast and cheap!
#
# Key concepts:
#   r (rank)        : Size of the LoRA adapter matrices. Higher = more capacity.
#                     8 or 16 is a good beginner value.
#   lora_alpha      : Scaling factor. Usually set to 2x the rank.
#   target_modules  : Which transformer layers to add LoRA to.
#                     q_proj and v_proj are the attention query/value projections.
#   lora_dropout    : Prevents overfitting (especially useful with small datasets).
#   bias="none"     : Don't train bias parameters (keeps it simple).

lora_config = LoraConfig(
    r=8,                                    # LoRA rank (adapter matrix size)
    lora_alpha=16,                          # Scaling factor
    target_modules=["q_proj", "v_proj"],    # Apply LoRA to attention layers
    lora_dropout=0.05,                      # Light dropout to prevent overfitting
    bias="none",                            # Don't modify bias parameters
    task_type=TaskType.CAUSAL_LM,           # We're doing causal language modeling
)

# Wrap the model with LoRA adapters
model = get_peft_model(model, lora_config)

# Print how many parameters we're actually training
model.print_trainable_parameters()
# Expected output: trainable params: ~4M (0.3%) out of 1.1B total
# This means we only train 4 million instead of 1.1 billion! 🎉


# ─────────────────────────────────────────────
# STEP 8: Set training arguments
# ─────────────────────────────────────────────
# What: Configures the training loop — how many steps, batch size, learning rate, etc.
# Why:  These settings control the quality and speed of training.
#
# Key settings for beginners:
#   num_train_epochs         : How many times to go through the whole dataset.
#                              3 epochs is a safe starting point.
#   per_device_train_batch_size : How many examples per GPU step.
#                                 Keep at 2 or 4 to avoid OOM errors.
#   gradient_accumulation_steps : Simulates a larger batch by accumulating gradients.
#                                 effective batch = batch_size × grad_accum = 2 × 4 = 8
#   learning_rate            : How fast the model learns. 2e-4 is standard for LoRA.
#   fp16=True                : Use 16-bit floats to save memory and speed up training.
#   logging_steps            : Print loss every N steps so you can see progress.
#   save_steps               : Save a checkpoint every N steps.
#   output_dir               : Where to save checkpoints.

training_args = TrainingArguments(
    output_dir="./finance-tinyllama-lora",  # Save checkpoints here
    num_train_epochs=3,                      # Train for 3 full passes over the data
    per_device_train_batch_size=2,           # 2 examples per GPU step (low memory)
    gradient_accumulation_steps=4,           # Effective batch size = 2 × 4 = 8
    learning_rate=2e-4,                      # Standard LoRA learning rate
    fp16=True,                               # Use float16 to save GPU memory
    logging_steps=10,                        # Log loss every 10 steps
    save_steps=50,                           # Save checkpoint every 50 steps
    save_total_limit=2,                      # Only keep last 2 checkpoints
    warmup_ratio=0.03,                       # Warm up LR for first 3% of steps
    lr_scheduler_type="cosine",              # Gradually reduce LR over training
    report_to="none",                        # Disable wandb/tensorboard logging
    dataloader_pin_memory=False,             # Avoids issues in some Colab environments
)
print("✅ Training arguments set")


# ─────────────────────────────────────────────
# STEP 9: Initialize the SFT Trainer and train
# ─────────────────────────────────────────────
# What: Creates a Supervised Fine-Tuning trainer and starts training.
# Why:  SFTTrainer from TRL is purpose-built for instruction fine-tuning.
#       It handles tokenization, batching, and training loops automatically.
#
# max_seq_length: Maximum token length of each training example.
#   Keep it at 512 for small datasets to avoid memory issues.
#   Our finance Q&A examples are short, so 256 would also work.
#
# Common mistake: Setting max_seq_length too high → Out of Memory (OOM) error.
#   Fix: Reduce max_seq_length or per_device_train_batch_size.

print("⏳ Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    dataset_text_field="text",   # Column name in our dataset that has the text
    max_seq_length=512,          # Max tokens per example
    packing=False,               # Don't pack multiple examples into one sequence
)

print("🚀 Starting fine-tuning... (this takes ~5-10 min on Colab T4 GPU)")
trainer.train()

# Expected training output:
# {'loss': 2.45, 'learning_rate': 0.0002, 'epoch': 0.5}
# {'loss': 1.87, 'learning_rate': 0.00015, 'epoch': 1.0}
# ... (loss should decrease over time — that means learning is happening!)
# Final loss around 0.5–1.0 is good for a small dataset like ours.


# ─────────────────────────────────────────────
# STEP 10: Save the fine-tuned LoRA adapter
# ─────────────────────────────────────────────
# What: Saves only the LoRA adapter weights (NOT the full model).
# Why:  The adapter is tiny (~20MB) vs the full model (~600MB).
#       The backend will load TinyLlama + merge in the adapter at startup.
#
# What gets saved:
#   adapter_config.json   → LoRA configuration
#   adapter_model.safetensors → The trained adapter weights

ADAPTER_SAVE_PATH = "./finance-tinyllama-lora/final-adapter"
trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)

print(f"✅ LoRA adapter saved to: {ADAPTER_SAVE_PATH}")
print("   Files saved:")
for f in os.listdir(ADAPTER_SAVE_PATH):
    size = os.path.getsize(os.path.join(ADAPTER_SAVE_PATH, f))
    print(f"   - {f} ({size / 1024:.1f} KB)")


# ─────────────────────────────────────────────
# STEP 11: Test the fine-tuned model
# ─────────────────────────────────────────────
# What: Generates a test response from the fine-tuned model.
# Why:  Quick sanity check before deploying to the API.

def generate_response(question, max_new_tokens=200):
    """Generates an answer to a finance question using the fine-tuned model."""
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"

    # Tokenize the prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",   # Return PyTorch tensors
        truncation=True,
        max_length=512,
    ).to(device)

    # Generate response
    with torch.no_grad():      # Don't compute gradients during inference (saves memory)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Maximum tokens to generate
            temperature=0.7,                 # Creativity: 0=deterministic, 1=creative
            do_sample=True,                  # Enable sampling (required for temperature)
            top_p=0.9,                       # Only consider top 90% probability tokens
            repetition_penalty=1.1,          # Penalize repeating the same words
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode generated tokens back to text
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (after "### Response:")
    if "### Response:" in full_output:
        response = full_output.split("### Response:")[-1].strip()
    else:
        response = full_output

    return response

# Test with a question from our dataset
print("\n" + "="*50)
print("🧪 Testing fine-tuned model:")
print("="*50)
test_question = "What is compound interest?"
answer = generate_response(test_question)
print(f"Q: {test_question}")
print(f"A: {answer}")

# Test with a question NOT in the dataset (generalization test)
print("\n" + "="*50)
print("🧪 Generalization test (unseen question):")
print("="*50)
unseen_question = "Why should I start investing early?"
answer2 = generate_response(unseen_question)
print(f"Q: {unseen_question}")
print(f"A: {answer2}")


# ─────────────────────────────────────────────
# STEP 12: Download the adapter from Colab
# ─────────────────────────────────────────────
# What: Zips and downloads the adapter so you can use it in the FastAPI backend.
# Why:  We need the saved adapter on your local machine for the backend server.
#
# Run this in a Colab cell:
# import shutil
# shutil.make_archive("finance_adapter", "zip", "./finance-tinyllama-lora/final-adapter")
# from google.colab import files
# files.download("finance_adapter.zip")
#
# Then unzip it into: backend/finance_adapter/


# ─────────────────────────────────────────────
# COMMON ERRORS & FIXES
# ─────────────────────────────────────────────
# 
# Error: CUDA out of memory
#   Fix: Reduce per_device_train_batch_size to 1
#        Reduce max_seq_length to 256
#        Make sure no other programs are using the GPU
#
# Error: ModuleNotFoundError: No module named 'bitsandbytes'
#   Fix: Run: !pip install bitsandbytes -q
#        Then restart the Colab runtime
#
# Error: ValueError: Tokenizer does not have a padding token
#   Fix: Already handled above — tokenizer.pad_token = tokenizer.eos_token
#
# Error: Loss is NaN (not a number)
#   Fix: Reduce learning rate to 1e-4
#        Make sure the dataset has no empty strings
#
# Error: Training loss is not decreasing
#   Fix: Increase num_train_epochs to 5
#        Check that your dataset is formatted correctly
#        Try increasing lora rank r to 16
