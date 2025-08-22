import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

DEBUG = False

def debug_print(debug_flag, *args, **kwargs):
    """Print debug messages if debug_flag is True."""
    if debug_flag:
        print(*args, **kwargs)

def merge_channels_for_output(channels: dict) -> str:
    """
    Merge channels in a defined order for legacy detectors or clean_output.
    Order: analysis -> commentary -> final
    """
    if not channels:
        return ""
    return "\n".join(str(channels[ch]) for ch in ["analysis", "commentary", "final"] if ch in channels)


def extract_content_from_raw(raw_text: str) -> Dict[str, str]:
    """
    Extract channels from a legacy raw_output string.
    Returns dict with 'final' key at minimum.
    """
    # Find all <|channel|>…<|message|> blocks
    channel_match = re.findall(
        r"<\|channel\|>(\w+)<\|message\|>(.*?)((?=<\|channel\|>)|$)", 
        raw_text, 
        re.DOTALL
    )

    if not channel_match:
        # No channels detected, return raw text as final
        return {"final": raw_text.strip()}

    channels = {}
    for name, content, _ in channel_match:
        # Strip control markers
        cleaned = re.split(r"<\|end\|>|<\|start\|>|<\|return\|>", content)[0].strip()
        channels[name] = cleaned

    # Ensure 'final' exists
    if "final" not in channels and channel_match:
        # pick the last channel as final
        channels["final"] = list(channels.values())[-1]

    return channels


def split_into_channels(output_or_dict) -> Dict[str, str]:
    """
    Extract channels from either:
        - a Harmony dict (modern)
        - a legacy raw string or Output object
    Returns dict: channel_name -> text
    """
    if isinstance(output_or_dict, dict):
        # Modern Harmony dict
        channels = {}
        for channel in ["final", "analysis", "commentary"]:
            channels[channel] = output_or_dict.get(channel, "")
        # Include extra keys
        for k, v in output_or_dict.items():
            if k not in channels:
                channels[k] = v
        return channels

    # Legacy path
    raw_text = getattr(output_or_dict, "raw_output", None) if hasattr(output_or_dict, "raw_output") else str(output_or_dict)
    if not raw_text:
        return {"final": ""}

    if isinstance(raw_text, dict):
        return raw_text

    return extract_content_from_raw(raw_text)


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