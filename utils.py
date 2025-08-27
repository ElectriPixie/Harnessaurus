# utils.py
import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from data_structures import Prompt, Output

DEBUG = False

def debug_print(debug_flag, *args, **kwargs):
    """Print debug messages if debug_flag is True."""
    if debug_flag:
        print(*args, **kwargs)



def unpack_textcontent(blob: str) -> str:
    """
    Parse a blob like "[TextContent(text='...')]" and return the inner text.
    If no match is found, return the original blob unchanged.
    """
    match = re.search(r"TextContent\(text='(.*?)'\)", blob, re.DOTALL)
    if match:
        return match.group(1)
    return blob


def merge_channels_for_output(channels: dict) -> str:
    """Merge channels in a defined order: analysis -> commentary -> final"""
    if not channels:
        return ""
    return "\n".join(str(channels[ch]) for ch in ["analysis", "commentary", "final"] if ch in channels)


def extract_content_from_raw(raw_text: str) -> Dict[str, str]:
    """
    Extract channels from a legacy raw_output string.
    Ensures a 'final' key is present.
    """
    channel_match = re.findall(
        r"<\|channel\|>(\w+)<\|message\|>(.*?)((?=<\|channel\|>)|$)",
        raw_text,
        re.DOTALL
    )

    if not channel_match:
        return {"final": raw_text.strip()}

    channels = {}
    for name, content, _ in channel_match:
        cleaned = re.split(r"<\|end\|>|<\|start\|>|<\|return\|>", content)[0].strip()
        channels[name] = cleaned

    if "final" not in channels and channel_match:
        channels["final"] = list(channels.values())[-1]

    return channels


# ----------------------------
# Legacy channel splitting
# ----------------------------
def split_into_channels_legacy(output) -> dict:
    """
    Legacy deterministic channel splitting.
    Extracts <|channel|>name<|message|>content from Output objects or raw strings.
    """
    if hasattr(output, "raw_output"):
        raw_text = output.raw_output
    elif isinstance(output, str):
        raw_text = output
    elif isinstance(output, dict) and "message" in output and "content" in output["message"]:
        raw_text = output["message"]["content"]
    else:
        return {}

    if not raw_text:
        return {}

    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(raw_text))
    if not matches:
        return {"final": raw_text.strip()}

    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        channels[channel_name] = raw_text[start:end].strip()

    if "final" not in channels:
        channels["final"] = list(channels.values())[-1]

    return channels


# ----------------------------
# Modern channel splitting (Harmony)
# ----------------------------
def split_into_channels(output_dict: dict) -> dict:
    """
    Modern Harmony-style channel extraction.
    Only supports dicts with keys like 'final', 'analysis', 'commentary'.
    """
    if not isinstance(output_dict, dict):
        raise TypeError(f"Expected dict for modern Harmony output, got {type(output_dict)}")

    channels = {}
    for channel in ["final", "analysis", "commentary"]:
        channels[channel] = output_dict.get(channel, "")
    for k, v in output_dict.items():
        if k not in channels:
            channels[k] = v

    return channels


def flip_negation(text: str) -> str:
    """
    Flip common negations in text (e.g., "don't" -> "do").
    """
    patterns = [
        (r"\bdo not\b", "do"), (r"\bdon't\b", "do"),
        (r"\bcan not\b", "can"), (r"\bcan't\b", "can"),
        (r"\bshould not\b", "should"), (r"\bshouldn't\b", "should"),
        (r"\bwill not\b", "will"), (r"\bwon't\b", "will")
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text
