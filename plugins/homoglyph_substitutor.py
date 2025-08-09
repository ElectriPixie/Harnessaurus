# plugins/homoglyph_substitutor.py
from plugin_base import PluginBase

class HomoglyphSubstitutor(PluginBase):
    HOMOGLYPHS = {
        'a': 'а',
        'e': 'е',
        'o': 'о',
        'i': 'і',
        'c': 'с',
        'y': 'у',
    }

    def process_prompt(self, prompt: str) -> str:
        return ''.join(self.HOMOGLYPHS.get(ch, ch) for ch in prompt)