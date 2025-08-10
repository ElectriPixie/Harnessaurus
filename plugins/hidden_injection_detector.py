from plugin_base import PluginBase
from pathlib import Path
from typing import List, Dict

# Default original smaller group
DEFAULT_CONTROL_CHARS = [
    '\u202A', '\u202B', '\u202C', '\u202D', '\u202E',
    '\u2066', '\u2067', '\u2068', '\u2069',
]

DEFAULT_ZERO_WIDTH_CHARS = ['\u200B', '\u200C', '\u200D', '\uFEFF']

class HiddenPromptInjectionDetector(PluginBase):
    def __init__(self, control_chars_file: str = "control_chars.txt", homoglyph_file: str = "homoglyphs.txt"):
        self.zero_width_chars = DEFAULT_ZERO_WIDTH_CHARS
        self.control_chars = self.load_control_chars(control_chars_file) or DEFAULT_CONTROL_CHARS
        self.homoglyphs = self.load_homoglyphs(homoglyph_file)

    def load_control_chars(self, filepath: str) -> List[str]:
        path = Path(filepath)
        if not path.is_file():
            return None
        chars = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Expect lines like '\u202A' -- convert to actual character
                try:
                    char = bytes(line, "utf-8").decode("unicode_escape")
                    chars.append(char)
                except Exception:
                    continue
        return chars if chars else None

    def load_homoglyphs(self, filepath: str) -> Dict[str, List[str]]:
        mapping = {}
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Homoglyph file not found: {filepath}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '#' in line:
                    line = line.split('#', 1)[0].strip()
                parts = line.split()
                if len(parts) < 2:
                    continue
                ascii_char = parts[0]
                glyphs = parts[1:]
                mapping[ascii_char] = glyphs
        return mapping

    def find_zero_width(self, text: str) -> List[str]:
        return [ch for ch in self.zero_width_chars if ch in text]

    def find_control_chars(self, text: str) -> List[str]:
        return [ch for ch in self.control_chars if ch in text]

    def find_homoglyphs(self, text: str) -> List[str]:
        found = []
        homoglyph_set = {g for glyphs in self.homoglyphs.values() for g in glyphs}
        for ch in text:
            if ch in homoglyph_set:
                found.append(ch)
        return found

    def process_output(self, prompt: str, output: str) -> Dict:
        z = self.find_zero_width(output)
        c = self.find_control_chars(output)
        h = self.find_homoglyphs(output)
        flagged = bool(z or c or h)
        return {
            'flagged': flagged,
            'zero_width_found': z,
            'control_chars_found': c,
            'homoglyphs_found': h,
        }