# plugin_base.py
from typing import List
from data_structures import Prompt, Output, Record

class PluginBase:
    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        """
        Default behavior: return the Prompt unchanged.
        Can be overridden by subclasses to modify prompts.
        """
        return prompt_obj

    def process_prompt_set(self, prompt_set: list[Prompt], **kwargs) -> list[Prompt]:
        """
        Apply process_prompt to each Prompt with flexible kwargs.
        """
        return [self.process_prompt(p, **kwargs) for p in prompt_set]

    def process_output(self, prompt_obj: Prompt, output_obj: Output, **kwargs) -> Output:
        """
        Analyze or modify an Output.
        Default: mark output as not flagged.
        Plugins should override to add analysis or flags.
        """
        if output_obj.analysis is None:
            output_obj.analysis = {}
        output_obj.analysis[self.__class__.__name__] = {'flagged': False}
        return output_obj

    def process_output_set(self, prompt_set: list[Prompt], output_set: list[Output], **kwargs) -> list[Output]:
        """
        Apply process_output to each (Prompt, Output) pair with flexible kwargs.
        """
        return [self.process_output(p, o, **kwargs) for p, o in zip(prompt_set, output_set)]

    def on_log(self, record_obj: Record) -> None:
        """
        Optional hook called after a Record is created.
        Can be overridden for logging, metrics, or side-effects.
        """
        pass