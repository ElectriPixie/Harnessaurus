# plugins/hidden_injection_detector.py
from plugin_base import PluginBase
from typing import Dict, List

ZERO_WIDTH_CHARS = ['\u200B', '\u200C', '\u200D', '\uFEFF']
CONTROL_CHARS = ['\u202A', '\u202B', '\u202C', '\u202D', '\u202E', '\u2066', '\u2067', '\u2068', '\u2069']
HOMOGLYPHS = {
    'a': ['а'],  # Cyrillic a
    'e': ['е'],
    'o': ['о'],
    'i': ['і'],
    'c': ['с'],
    'p': ['р'],
    'y': ['у'],
    'x': ['х'],
}

def find_zero_width(text: str) -> List[str]:
    return [ch for ch in ZERO_WIDTH_CHARS if ch in text]

def find_control_chars(text: str) -> List[str]:
    return [ch for ch in CONTROL_CHARS if ch in text]

def find_homoglyphs(text: str) -> List[tuple]:
    found = []
    for ascii_c, glyphs in HOMOGLYPHS.items():
        for g in glyphs:
            if g in text:
                found.append((ascii_c, g))
    return found

class HiddenPromptInjectionDetector(PluginBase):
    def __init__(self):
        pass

    def process_output(self, prompt: str, output: str) -> Dict:
        z = find_zero_width(output)
        c = find_control_chars(output)
        h = find_homoglyphs(output)
        flagged = bool(z or c or h)
        return {
            'flagged': flagged,
            'zero_width_found': z,
            'control_chars_found': c,
            'homoglyphs_found': h,
        }

    def process_batch(self, prompts: List[str], outputs: List[str]) -> List[Dict]:
        return [self.process_output(p, o) for p, o in zip(prompts, outputs)]
