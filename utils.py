# utils.py
import re

def debug_print(debug_flag, *args, **kwargs):
    if debug_flag:
        print(*args, **kwargs)

def split_into_channels(output):
    """Extract <|channel|>content<|message|> sections from Output object."""
    if not hasattr(output, 'raw_output') or not output.raw_output:
        return {}
    pattern = re.compile(r"<\|channel\|>(\w+)<\|message\|>")
    matches = list(pattern.finditer(output.raw_output))
    channels = {}
    for i, match in enumerate(matches):
        channel_name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output.raw_output)
        channels[channel_name] = output.raw_output[start:end].strip()
    return channels

def flip_negation(text: str) -> str:
    """Flip common negations in text."""
    patterns = [
        (r"\bdo not\b", "do"), (r"\bdon't\b", "do"),
        (r"\bcan not\b", "can"), (r"\bcan't\b", "can"),
        (r"\bshould not\b", "should"), (r"\bshouldn't\b", "should"),
        (r"\bwill not\b", "will"), (r"\bwon't\b", "will")
    ]
    for pat, repl in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    if DEBUG:
        print(f"[flip_negation] Result: {text[:100]}...")
    return text
