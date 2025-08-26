import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging
import time
import uuid
import argparse
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Role,
    Message,
    Conversation,
)

import pprint

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style Harmony server")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--deterministic", choices=["on","off"], default="on")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repetition_control", choices=["on","off"], default="on")
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
parser.add_argument("--port", type=int, default=6589)
args = parser.parse_args()

DETERMINISTIC = args.deterministic.lower() == "on"
REPETITION_CONTROL = args.repetition_control.lower() == "on"
DEBUG_MODEL_DATA = False  # Set to True if you want hidden_states and attentions

# --- Deterministic Setup ---
if DETERMINISTIC:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

# --- Logging ---
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/generation_{timestamp}.log"
logger = logging.getLogger("harmony_server")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

logger.info(f"Starting Harmony server with model: {args.model_id}")

# --- FastAPI app ---
app = FastAPI()

# --- Load Model and Encoding ---
logger.info(f"Loading model {args.model_id}...")
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
device = next(model.parameters()).device
logger.info(f"Model loaded on device {device}")

# --- Utility Functions ---
INSTRUCTIONS = "You are an AI assistant serving an insane mad scientist."

def pretty_log(title, obj):
    """Human-readable logging helper."""
    formatted = pprint.pformat(obj, width=120)
    logger.info(f"\n=== {title} ===\n{formatted}\n=== End {title} ===")

def conversation_to_string(convo):
    """Convert a Harmony Conversation into a single prompt string."""
    messages_text = []
    for msg in convo.messages:
        role = getattr(msg, "role", "unknown")
        content = getattr(msg, "content", str(msg))
        messages_text.append(f"[{role}] {content}")
    return "\n".join(messages_text)

def generate_text(messages, max_tokens):
    """Generate text synchronously with readable debug logs."""
    convo = Conversation.from_messages(
        [Message.from_role_and_content(Role.SYSTEM, INSTRUCTIONS)] +
        [Message.from_role_and_content(Role.USER if m["role"]=="user" else Role.ASSISTANT, m["content"]) for m in messages]
    )
    context = conversation_to_string(convo)
    pretty_log("Prompt Sent to Model", context)

    # Tokenize
    inputs = tokenizer(context, return_tensors="pt").to(model.device)
    logger.info(f"Input tensor shape: {inputs.input_ids.shape}")
    logger.info(f"Sample token IDs: {inputs.input_ids[0][:20].tolist()}")

    # --- Generation ---
    with torch.no_grad():
        do_sample_flag = args.do_sample and not DETERMINISTIC
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample_flag,
            "return_dict_in_generate": True,
            "output_hidden_states": DEBUG_MODEL_DATA,
            "output_attentions": DEBUG_MODEL_DATA
        }
        if do_sample_flag:
            generate_kwargs.update({
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p
            })
        if REPETITION_CONTROL:
            if args.no_repeat_ngram_size > 0:
                generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
            if args.repetition_penalty != 1.0:
                generate_kwargs["repetition_penalty"] = args.repetition_penalty

        pretty_log("Generation Args", generate_kwargs)
        outputs = model.generate(**generate_kwargs, **inputs)

    prev_len = inputs.input_ids.shape[1]
    new_tokens = outputs.sequences[0][prev_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    pretty_log("Model Output", text)

    # Optionally log hidden states/attentions
    if DEBUG_MODEL_DATA:
        all_hidden_states = [h.cpu().tolist() for h in outputs.hidden_states]
        all_attentions = [a.cpu().tolist() for a in outputs.attentions]
        pretty_log("Hidden States", all_hidden_states)
        pretty_log("Attentions", all_attentions)

    return text

# --- API Schema ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = args.max_tokens

# --- Routes ---
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    request_id = str(uuid.uuid4())
    logger.info(f"[API] Incoming request | Request ID: {request_id} | IP: {request.client.host}")
    pretty_log("Request Messages", req.messages)

    try:
        final_text = generate_text(req.messages, req.max_tokens)
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(final_text.split()), "total_tokens": len(final_text.split())}
        }
        pretty_log("API Response", response)
        return JSONResponse(response)
    except Exception as e:
        logger.exception(f"[API] Error generating response for Request ID: {request_id}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/v1/version")
async def version():
    return {"version": "0.6.0-hf-template", "model_id": args.model_id, "deterministic": "Yes" if DETERMINISTIC else "No"}

# --- Run Server ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
