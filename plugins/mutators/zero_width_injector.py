import random
import codecs
from pathlib import Path
from plugin_base import MutatorPlugin
from data_structures import Prompt

class ZeroWidthInjector(MutatorPlugin):
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

    def __init__(self, invisible_chars_file: str = None, intensity: float = 0.3, seed: int = None):
        self.intensity = intensity
        self.INVISIBLE_CHARS = self.DEFAULT_INVISIBLE_CHARS.copy()
        if invisible_chars_file:
            try:
                self.INVISIBLE_CHARS = self.load_invisible_chars(invisible_chars_file)
            except FileNotFoundError:
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
                line = line.split('#', 1)[0].strip()
                if line:
                    chars.append(codecs.decode(line, 'unicode_escape'))
        return chars

    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        new_prompts = []

        for text in prompt_obj.prompt_list:
            result = []
            for ch in text:
                result.append(ch)
                if self.INVISIBLE_CHARS and random.random() < self.intensity:
                    result.append(random.choice(self.INVISIBLE_CHARS))
            new_prompts.append("".join(result))

        prompt_obj.prompt_list = new_prompts
        return prompt_obj
