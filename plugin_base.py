# plugin_base.py
from typing import List, Union
from data_structures import Prompt, PromptSet, Output, Record

class PluginBase:
    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        """
        Default behavior: return the Prompt unchanged.
        Can be overridden by subclasses to modify prompts.
        """
        return prompt_obj

    def process_prompt_set(
        self,
        prompt_set: Union[List[Prompt], PromptSet],
        **kwargs
    ) -> PromptSet:
        """
        Apply process_prompt to each Prompt in a PromptSet or list.
        Returns a PromptSet wrapping all results.
        """
        # Convert list to PromptSet if needed
        if isinstance(prompt_set, list):
            prompt_set = PromptSet(prompts=prompt_set)

        result_set = PromptSet(output_type="multi")
        for p in prompt_set:
            result_set.add_prompt(self.process_prompt(p, **kwargs))

        # Preserve tags if the input was a PromptSet
        if hasattr(prompt_set, "tags"):
            result_set.tags.update(prompt_set.tags)

        return result_set

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

    def process_output_set(
        self,
        prompt_set: Union[List[Prompt], PromptSet],
        output_set: List[Output],
        **kwargs
    ) -> List[Output]:
        """
        Apply process_output to each (Prompt, Output) pair with flexible kwargs.
        Works with both list[Prompt] and PromptSet.
        """
        # Convert PromptSet to list if needed
        prompts = list(prompt_set) if isinstance(prompt_set, PromptSet) else prompt_set
        return [self.process_output(p, o, **kwargs) for p, o in zip(prompts, output_set)]

    def on_log(self, record_obj: Record) -> None:
        """
        Optional hook called after a Record is created.
        Can be overridden for logging, metrics, or side-effects.
        """
        pass