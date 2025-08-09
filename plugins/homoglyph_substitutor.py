# plugins/homoglyph_substitutor.py
import random
from plugin_base import PluginBase

class HomoglyphSubstitutor(PluginBase):
    def __init__(self, mapping_file='homoglyphs.txt'):
        self.homoglyph_map = self.load_homoglyphs(mapping_file)

    def load_homoglyphs(self, filepath):
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                base_char = parts[0]
                homoglyphs = parts[1:]
                if homoglyphs:
                    mapping[base_char] = homoglyphs
        return mapping

    def process_prompt(self, prompt: str) -> str:
        result_chars = []
        for ch in prompt:
            if ch in self.homoglyph_map:
                # Randomly pick one homoglyph for substitution
                result_chars.append(random.choice(self.homoglyph_map[ch]))
            else:
                result_chars.append(ch)
        return ''.join(result_chars)
