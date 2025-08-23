import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# --- FastAPI endpoints ---
from fastapi.responses import StreamingResponse

import logging
import threading
from queue import Queue, Empty
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import time
import uuid
import argparse
import math
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent
)

# --- CLI arguments ---
parser = argparse.ArgumentParser(description="OpenAI-style Harmony server with paginated chunks")
parser.add_argument("--model_id", type=str, default="/data/AI/Models/gpt-oss-20b")
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--chunk_size", type=int, default=512)
parser.add_argument("--num_threads", type=int, default=4)
parser.add_argument("--multi_chunk", action="store_true", default=False)
parser.add_argument("--chunk_streaming", action="store_true", default=False)
parser.add_argument("--strip_legacy_format", action="store_true", default=True)
parser.add_argument("--port", type=int, default=6589)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true", default=False)
parser.add_argument("--deterministic", choices=["on", "off"], default="on")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--repetition_control", choices=["on", "off"], default="on")
parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

DETERMINISTIC = args.deterministic.lower() == "on"
REPETITION_CONTROL = args.repetition_control.lower() == "on"

# --- Deterministic setup ---
if DETERMINISTIC:
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    logging.info(f"Deterministic mode ON | Seed: {args.seed}")
else:
    torch.use_deterministic_algorithms(False)
    logging.info("Deterministic mode OFF | Using PyTorch default RNG")

# --- Logging ---
os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join("logs", f"generation_{timestamp}.log")
logger = logging.getLogger("harmony_server")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [Thread %(threadName)s] %(message)s"))
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

logger.debug(f"Starting Harmony server with args: {args}")
logger.info(f"Starting Harmony server with model: {args.model_id}")

# --- FastAPI app ---
app = FastAPI()

# --- Load model ---
logger.info(f"Loading model {args.model_id}...")
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
tokenizer = AutoTokenizer.from_pretrained(args.model_id)
model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype="auto", device_map="auto")
logger.info(f"Model loaded on device {next(model.parameters()).device}")

# --- Threading + IPC ---
job_queue = Queue()
results = {}                # job_id -> final result dict
# Per-request chunk queues for streaming: request_id -> queue.Queue()
stream_queues = {}
lock = threading.Lock()

# --- Helper ---
def flatten_entries(entries):
    flat_list = []
    for e in entries:
        if isinstance(e, list):
            flat_list.extend(flatten_entries(e))
        elif hasattr(e, "content"):
            flat_list.append(str(e.content))
        else:
            flat_list.append(str(e))
    return flat_list

def generate_harmony_chunk(input_ids, attention_mask, tokens_to_request, stop_token_ids, request_id):
    """Generates a single chunk of text from the model, returns (chunk_text, completion_ids, reached_eos)."""
    do_sample_flag = args.do_sample and not DETERMINISTIC
    generate_kwargs = {
        "max_new_tokens": tokens_to_request,
        "do_sample": do_sample_flag,
        "attention_mask": attention_mask,
        "eos_token_id": stop_token_ids
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

    logger.debug(f"[Request {request_id}] Generating chunk | kwargs={generate_kwargs}")
    logger.debug(f"[Request {request_id}] Input IDs: {input_ids.tolist()[:20]}... total {input_ids.numel()} tokens")

    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, **generate_kwargs)

    completion_ids = outputs[0, input_ids.shape[1]:] if outputs is not None else torch.tensor([], device=model.device, dtype=torch.long)

    # Strip legacy tokens
    if args.strip_legacy_format:
        LEGACY_TOKENS = set(range(200000, 200008))
        if completion_ids.numel():
            filtered = [t for t in completion_ids.tolist() if t not in LEGACY_TOKENS]
            completion_ids = torch.tensor(filtered, device=model.device, dtype=outputs.dtype)

    reached_eos = False
    if len(stop_token_ids):
        reached_eos = any(int(t) in stop_token_ids for t in completion_ids.tolist())

    # --- Robust parsing ---
    chunk_text = ""
    if completion_ids.numel() > 0:
        try:
            assistant_entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
            parsed = " ".join(flatten_entries(assistant_entries)) if assistant_entries else ""
            if parsed:
                chunk_text = parsed
            else:
                # Fallback: raw decode if Harmony yielded nothing
                chunk_text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)
        except Exception as e:
            logger.exception(f"[Request {request_id}] Failed to parse assistant entries: {e}")
            # Fallback: raw decode on parse failure
            chunk_text = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=True)

    logger.debug(f"[Request {request_id}] generate_harmony_chunk -> completion_ids_len={completion_ids.numel()} chunk_text_len={len(chunk_text)}")
    return chunk_text, completion_ids, reached_eos

# --- Worker + robust streaming via per-request Queue ---
def model_worker():
    while True:
        job = job_queue.get()
        if job is None:
            logger.debug("Worker received shutdown signal")
            break

        # Defensive log
        try:
            logger.debug(f"[Worker] Retrieved job from queue | Job tuple length={len(job)} | Request ID: {job[4]}")
        except Exception:
            logger.debug(f"[Worker] Retrieved job (malformed tuple): {job}")

        # Unpack job safely
        if len(job) == 6:
            job_id, messages, max_tokens_param, multi_chunk, request_id, chunk_streaming = job
            chunk_size = min(max_tokens_param or args.max_tokens, args.chunk_size)
        else:
            job_id, messages, max_tokens_param, multi_chunk, request_id, chunk_streaming, chunk_size_param = job
            chunk_size = min(chunk_size_param, max_tokens_param or args.max_tokens)

        logger.debug(f"[Worker] Picked up job {job_id} | Request ID: {request_id} | max_tokens={max_tokens_param} | chunk_size={chunk_size}")

        start_time = time.time()
        full_text = ""
        tokens_generated = 0

        try:
            # Build conversation
            convo_messages = [
                Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                Message.from_role_and_content(Role.DEVELOPER,
                                              DeveloperContent.new().with_instructions("Respond helpfully and safely"))
            ]
            for msg in messages:
                convo_messages.append(Message.from_role_and_content(Role.USER, msg["content"]))

            convo = Conversation.from_messages(convo_messages)
            prefill_ids_list = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
            while prefill_ids_list and prefill_ids_list[-1] == 200007:
                prefill_ids_list.pop()
            if not prefill_ids_list or prefill_ids_list[-1] != 200006:
                prefill_ids_list.append(200006)

            input_ids = torch.tensor([prefill_ids_list], device=model.device)
            attention_mask = torch.ones_like(input_ids)

            max_iterations = math.ceil((max_tokens_param or args.max_tokens) / chunk_size) if multi_chunk else 1

            for i in range(max_iterations):
                is_last = (i == max_iterations - 1)
                tokens_to_request = min(chunk_size, (max_tokens_param or args.max_tokens) - tokens_generated)
                stop_token_ids = encoding.stop_tokens_for_assistant_actions() if is_last else []

                logger.debug(f"[Worker] Iteration {i} | Requesting {tokens_to_request} tokens | is_last={is_last}")

                # Generate chunk
                chunk_text, completion_ids, reached_eos = generate_harmony_chunk(
                    input_ids, attention_mask, tokens_to_request, stop_token_ids, request_id
                )

                # Ensure we have something to stream where possible
                if not chunk_text and completion_ids is not None and completion_ids.numel() > 0:
                    raw_attempt = tokenizer.decode(completion_ids.tolist(), skip_special_tokens=False)
                    if raw_attempt:
                        logger.warning(f"[Worker] Harmony parse empty; using raw decode | len={len(raw_attempt)} | preview={repr(raw_attempt[:80])}")
                        chunk_text = raw_attempt
                    else:
                        logger.warning(f"[Worker] Both Harmony parse and raw decode empty | completion_ids_len={completion_ids.numel()}")

                logger.debug(f"[Worker] Decoded chunk_text len={len(chunk_text)} | preview={repr((chunk_text or '')[:80])}")

                # Put chunk into per-request stream queue if present
                with lock:
                    q = stream_queues.get(request_id)
                if q is not None:
                    try:
                        # Put the chunk (may be empty string) so the generator sees an event
                        q.put(chunk_text or "")
                        logger.debug(f"[Worker] put chunk into stream_queue for request={request_id} (+{len(chunk_text or '')} chars)")
                    except Exception as exc:
                        logger.exception(f"[Worker] Failed to put chunk into stream_queue: {exc}")

                # Always accumulate final text for blocking responses
                full_text += (chunk_text or "")
                tokens_generated += len((chunk_text or "").split())

                # Prepare next input for iterative generation
                try:
                    if completion_ids is not None and completion_ids.numel() > 0:
                        input_ids = torch.cat([input_ids, completion_ids.unsqueeze(0)], dim=1)
                        attention_mask = torch.ones_like(input_ids)
                except Exception as exc:
                    logger.exception(f"[Worker] Failed to append completion_ids to input_ids: {exc}")

                # Break early if streaming and EOS reached
                if chunk_streaming and reached_eos:
                    finish_reason = "stop"
                    logger.debug(f"[Worker] EOS detected at iteration {i}")
                    break
            else:
                finish_reason = "length"

            duration = time.time() - start_time

            # Store final result
            with lock:
                results[job_id] = {
                    "text": full_text.strip(),
                    "finish_reason": finish_reason,
                    "duration": duration,
                    "prompt_length": sum(len(m["content"].split()) for m in messages),
                    "completion_length": len(full_text.split()),
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk
                }

            logger.debug(f"[Worker] Finished job {job_id} | Duration: {duration:.2f}s | Tokens: {tokens_generated} | Finish reason: {finish_reason}")

            # If a stream queue exists for this request, place a sentinel to signal completion.
            with lock:
                q = stream_queues.get(request_id)
            if q is not None:
                try:
                    q.put(None)  # sentinel to indicate done
                    logger.debug(f"[Worker] put sentinel None into stream_queue for request={request_id}")
                except Exception as exc:
                    logger.exception(f"[Worker] Failed to put sentinel into stream_queue: {exc}")

        except Exception as e:
            logger.exception(f"[Worker] ERROR Job {job_id} | Request ID: {request_id}")
            with lock:
                results[job_id] = {
                    "text": f"Error: {e}",
                    "finish_reason": "error",
                    "duration": 0,
                    "prompt_length": sum(len(m["content"].split()) for m in messages),
                    "completion_length": 0,
                    "device": str(model.device),
                    "request_id": request_id,
                    "multi_chunk": multi_chunk
                }
            # If streaming, ensure client gets sentinel on error
            with lock:
                q = stream_queues.get(request_id)
            if q is not None:
                try:
                    q.put(None)
                except Exception:
                    pass
        finally:
            job_queue.task_done()

# --- Start workers ---
for i in range(args.num_threads):
    threading.Thread(target=model_worker, daemon=True, name=f"Worker-{i}").start()
    logger.debug(f"Started worker thread Worker-{i}")

# --- Job submission ---
def submit_generation(messages, max_tokens=None, multi_chunk=None, request_id=None,
                      chunk_streaming=False, chunk_size=None):
    """Submit job and optionally stream output.

    For streaming requests, this returns a **generator** that yields chunk strings as they arrive.
    For blocking requests, it returns the final result dict.
    """
    max_tokens = max_tokens or args.max_tokens
    multi_chunk = multi_chunk if multi_chunk is not None else args.multi_chunk
    chunk_size = chunk_size or args.chunk_size

    with lock:
        job_id = len(results)
        logger.debug(f"Submitting job {job_id} | Request ID: {request_id}")

    # Ensure a per-request queue exists for streaming requests
    if chunk_streaming:
        with lock:
            # create a new queue for this request id (thread-safe queue.Queue)
            q = Queue()
            stream_queues[request_id] = q
            logger.debug(f"Created stream_queue for request={request_id}")

    # Put the job on the worker queue
    job_queue.put((job_id, messages, max_tokens, multi_chunk, request_id, chunk_streaming, chunk_size))
    logger.debug(f"Job {job_id} put in queue (len={job_queue.qsize()}) | Request ID: {request_id}")

    if not chunk_streaming:
        # Blocking: wait for job result
        while True:
            with lock:
                if job_id in results:
                    return results.pop(job_id)
            time.sleep(0.01)
    else:
        # Streaming: return a generator that yields items from the per-request queue
        def stream_generator():
            q_local = None
            with lock:
                q_local = stream_queues.get(request_id)
            if q_local is None:
                logger.error(f"[stream_generator] No stream queue found for request {request_id}; exiting")
                return

            try:
                while True:
                    try:
                        # block until a chunk or sentinel arrives
                        item = q_local.get(timeout=30.0)  # 30s timeout to avoid indefinite blocking
                    except Empty:
                        logger.debug(f"[stream_generator] timeout waiting for chunk for request {request_id}")
                        # Check if job finished silently
                        with lock:
                            done = job_id in results
                        if done:
                            break
                        else:
                            continue

                    # sentinel -> done
                    if item is None:
                        logger.debug(f"[stream_generator] received sentinel for request {request_id}; ending stream")
                        break

                    # yield whatever string we got (may be "")
                    if item:
                        yield item

                # final flush: if worker produced final aggregate but didn't stream it (rare), send it
                with lock:
                    # if results ready, send any remaining text that wasn't streamed
                    if job_id in results:
                        final = results[job_id]["text"]
                        if final:
                            yield final
            finally:
                # cleanup queue
                with lock:
                    stream_queues.pop(request_id, None)
                logger.debug(f"[stream_generator] cleaned up stream_queue for request {request_id}")

        return stream_generator()

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = args.max_tokens
    multi_chunk: bool = None
    chunk_size: int = None
    chunk_stream: bool = False  # switch between blocking/streaming

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    request_id = str(uuid.uuid4())
    multi_chunk = req.multi_chunk if req.multi_chunk is not None else args.multi_chunk
    chunk_size = req.chunk_size if req.chunk_size else args.chunk_size
    use_stream = req.chunk_stream if req.chunk_stream is not None else args.chunk_streaming

    logger.info(f"[API] Incoming request | Request ID: {request_id} | IP: {request.client.host} | Multi-chunk: {multi_chunk} | Stream: {use_stream} | Chunk size: {chunk_size}")
    if req.messages:
        logger.info(f"[API] Last user message preview: {req.messages[-1]['content'][:200]}")

    result_or_gen = submit_generation(
        req.messages,
        max_tokens=req.max_tokens,
        multi_chunk=multi_chunk,
        request_id=request_id,
        chunk_streaming=use_stream,
        chunk_size=chunk_size
    )

    if not use_stream:
        result = result_or_gen  # dict
        logger.info(f"[API] Response ready | Request ID: {request_id} | Tokens: {result['completion_length']} | Finish reason: {result['finish_reason']}")
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": result["finish_reason"],
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": result["prompt_length"],
                "completion_tokens": result["completion_length"],
                "total_tokens": result["prompt_length"] + result["completion_length"],
            },
        }
    else:
        logger.info(f"[API] Streaming response started | Request ID: {request_id}")
        # StreamingResponse can accept a generator (sync) which we provide
        return StreamingResponse(result_or_gen, media_type="text/plain")

@app.get("/v1/version")
async def version():
    return {
        "version": "0.6.0-debug-queue-stream",
        "model_id": args.model_id,
        "harmony_mode": "OpenAI-Harmony",
        "deterministic": "Yes" if DETERMINISTIC else "No"
    }

if __name__ == "__main__":
    logger.info(f"Starting server on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)