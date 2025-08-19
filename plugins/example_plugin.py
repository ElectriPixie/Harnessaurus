# plugins/example_plugin.py
from plugin_base import PluginBase
from typing import Dict, Any

class ExamplePlugin(PluginBase):
    def __init__(self, greet: str = "hello"):
        self.greet = greet

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        """
        Always returns metadata for logging purposes.
        """
        return {
            'flagged': False,
            'meta': f"{self.greet} (len_output={len(output)})"
        }

    def process_prompt(self, prompt: str) -> str:
        # This plugin does not modify the prompt
        return prompt