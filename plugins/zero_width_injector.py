# plugins/zero_width_injector.py
from plugin_base import PluginBase

class ZeroWidthInjector(PluginBase):
    ZERO_WIDTH_SPACE = '\u200b'

    def process_prompt(self, prompt: str) -> str:
        chars = list(prompt)
        for i in range(3, len(chars), 4):
            chars.insert(i, self.ZERO_WIDTH_SPACE)
        return ''.join(chars)
