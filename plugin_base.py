# plugin_base.py
from typing import Dict, Any

class PluginBase:
    def process_prompt(self, prompt: str, mutation_index: int = 0) -> str:
        return prompt

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        return {'flagged': False}

    def on_log(self, record: Dict[str, Any]) -> None:
        pass