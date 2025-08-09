# plugin_base.py
from typing import Dict, Any

class PluginBase:
    def process_prompt(self, prompt: str) -> str:
        return prompt

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        return {'flagged': False}

    def process_batch(self, prompts: list, outputs: list) -> list:
        # By default, fallback to per-example processing
        return [self.process_output(p, o) for p, o in zip(prompts, outputs)]

    def on_log(self, record: Dict[str, Any]) -> None:
        pass
