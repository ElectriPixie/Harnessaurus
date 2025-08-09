import random
import codecs
from plugin_base import PluginBase
from pathlib import Path

class ZeroWidthInjector(PluginBase):
    DEFAULT_INVISIBLE_CHARS = [
        '\u200b',  # ZERO WIDTH SPACE
        '\u200c',  # ZERO WIDTH NON-JOINER
        '\u200d',  # ZERO WIDTH JOINER
        '\ufeff',  # ZERO WIDTH NO-BREAK SPACE (BOM)
        '\u200e',  # LEFT-TO-RIGHT MARK
        '\u200f',  # RIGHT-TO-LEFT MARK
        '\u2060',  # WORD JOINER
        '\u2063',  # INVISIBLE SEPARATOR
        '\u2064',  # INVISIBLE PLUS
    ]

    def __init__(self, invisible_chars_file: str = None, intensity=0.3, seed=None):
        """
        invisible_chars_file: optional path to a file listing invisible characters, one per line,
                             comments start with # and are ignored.
        intensity: float (0 to 1) controlling insertion frequency of invisible chars.
        seed: Optional seed for reproducibility.
        """
        self.intensity = intensity
        self.INVISIBLE_CHARS = self.DEFAULT_INVISIBLE_CHARS
        if invisible_chars_file:
            try:
                self.INVISIBLE_CHARS = self.load_invisible_chars(invisible_chars_file)
            except FileNotFoundError:
                # Fallback to default list if file not found
                pass
        if seed is not None:
            random.seed(seed)

    def load_invisible_chars(self, filepath: str) -> list[str]:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Invisible chars file not found: {filepath}")
        chars = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()  # Remove comments and whitespace
                if not line:
                    continue
                # Decode escaped unicode sequences like \u200b to actual char
                decoded_char = codecs.decode(line, 'unicode_escape')
                chars.append(decoded_char)
        return chars

    def process_prompt(self, prompt: str) -> str:
        result = []
        for ch in prompt:
            result.append(ch)
            if random.random() < self.intensity and self.INVISIBLE_CHARS:
                inv_char = random.choice(self.INVISIBLE_CHARS)
                result.append(inv_char)
        return ''.join(result)
