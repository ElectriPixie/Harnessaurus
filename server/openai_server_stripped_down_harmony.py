import os
import re
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
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
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
DEBUG_MODEL_DATA = False

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
    formatted = pprint.pformat(obj, width=120)
    logger.info(f"\n=== {title} ===\n{formatted}\n=== End {title} ===")

def parse_role_tokened_output(raw_text):
    """
    Parses <|start|>role<|message|>content<|end|> blocks into a list of dicts.
    For assistant messages with channels, include 'channel' key.
    """
    messages = []
    # Pattern matches: <|start|>role[<|channel|>optional_channel]<|message|>content<|end|>
    pattern = re.compile(
        r"<\|start\|>(?P<role>\w+)(?:<\|channel\|>(?P<channel>\w+))?<\|message\|>(?P<content>.*?)<\|end\|>",
        re.DOTALL
    )
    for match in pattern.finditer(raw_text):
        msg = {
            "role": match.group("role"),
            "message": match.group("content").strip()
        }
        if match.group("channel"):
            msg["channel"] = match.group("channel")
        messages.append(msg)
    return messages

def flatten_content(c):
    if isinstance(c, list):
        return " ".join([flatten_content(i) for i in c])
    if hasattr(c, "text"):
        return c.text  # just the raw text
    return str(c)

def format_messages_with_tokens(convo_msgs, entries):
    output_lines = []

    # System + Developer + User messages
    for msg in convo_msgs:
        role_enum = getattr(msg, "role", None) or Role.SYSTEM
        role_str = role_enum.name.lower() if isinstance(role_enum, Role) else "system"
        output_lines.append(f"<|start|>{role_str}<|message|>{flatten_content(msg.content)}<|end|>")

    # Assistant / Analysis messages
    for msg in entries:
        role_enum = getattr(msg, "role", None) or Role.ASSISTANT
        role_str = role_enum.name.lower() if isinstance(role_enum, Role) else "assistant"

        # Only add <|channel|> if it exists
        if hasattr(msg, "channel") and msg.channel:
            output_lines.append(f"<|start|>{role_str}<|channel|>{msg.channel}<|message|>{flatten_content(msg.content)}<|end|>")
        else:
            output_lines.append(f"<|start|>{role_str}<|message|>{flatten_content(msg.content)}<|end|>")

    return "\n".join(output_lines)

def generate_text_harmony(messages, max_tokens):
    """
    Generate text with Harmony conversation, producing role-tokened output and channels.
    """
    # --- Build conversation ---
    convo_msgs = [
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(INSTRUCTIONS)
        ),
        *[
            Message.from_role_and_content(Role.USER, m["content"])
            for m in messages
            if m["role"] == "user"  # skip assistant messages
        ]
    ]
    convo = Conversation.from_messages(convo_msgs)

    # --- Log input conversation ---
    pretty_log("Prompt Sent to Model", [str(msg) for msg in convo_msgs])
    logger.info(f"Number of messages in convo: {len(convo_msgs)}")

    # --- Render conversation tokens ---
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    stop_ids = encoding.stop_tokens_for_assistant_actions()
    logger.info(f"Prefill token length: {len(prefill_ids)}")
    logger.info(f"Stop token IDs: {stop_ids}")

    # --- Convert token IDs to tensor ---
    input_ids = torch.tensor([prefill_ids], dtype=torch.long, device=device)

    # --- Generate model output ---
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=stop_ids
        )
    logger.info(f"Generated {outputs.shape[1] - len(prefill_ids)} new tokens")

    # --- Parse completion tokens ---
    completion_ids = outputs[0][len(prefill_ids):]
    entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)

    # --- Build raw text and channels using fixed formatter ---
    raw_text = format_messages_with_tokens(convo_msgs, entries)
    assistant_channels = []
    for msg in entries:
            if hasattr(msg, "channel") and msg.channel:
                channel_name = msg.channel
                assistant_channels.append({"channel": channel_name, "content": str(msg.content)})

    pretty_log("Raw Model Output", raw_text)
    pretty_log("Asisstant Channels", assistant_channels)

    return raw_text, assistant_channels

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
        raw_text, channels = generate_text_harmony(req.messages, req.max_tokens)
        parsed_messages = parse_role_tokened_output(raw_text)
        # Pretty print for logging
        pretty_log("Parsed Messages", parsed_messages)                        
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "raw_text": raw_text,
            "channels": channels,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": raw_text}, "finish_reason": "stop"}
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(raw_text.split()),
                "total_tokens": len(raw_text.split())
            }
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
