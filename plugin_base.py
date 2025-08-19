# plugin_base.py
from typing import Dict, Any

class PluginBase:
    def process_prompt(self, prompt: str, mutation_index: int = 0, ret_list: bool = False, with_context: bool = False) -> str:
        # Default behavior, can be overridden by subclasses
        return prompt

    def process_prompt_set(
        self,
        prompt_set: list[str],
        mutation_index: int = 0,
        ret_list: bool = False,
        with_context: bool = False
    ) -> list[str]:
        # Apply process_prompt to each prompt in the set
        return [
            self.process_prompt(prompt, mutation_index=mutation_index, ret_list=ret_list, with_context=with_context)
            for prompt in prompt_set
        ]

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        return {'flagged': False}

    def process_output_set(self, prompt_set: list[str], output_set: list[str]) -> list[Dict[str, Any]]:
        results = []
        for prompt, output in zip(prompt_set, output_set):
            results.append(self.process_output(prompt, output))
        return results

    def on_log(self, record: Dict[str, Any]) -> None:
        pass