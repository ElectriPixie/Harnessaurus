# plugin_base.py
from typing import List, Union
from data_structures import Prompt, PromptSet, Output, Record, OutputType

class BasePlugin:
    """Minimal plugin functionality: shared hooks and metadata."""
    def on_log(self, record_obj: Record) -> None:
        """Optional hook called after a Record is created."""
        pass

class DetectorPlugin(BasePlugin):
    """Plugins that analyze outputs."""
    def process_output(self, prompt_obj: Prompt, output_obj: Output, **kwargs) -> Output:
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
        prompts = list(prompt_set) if isinstance(prompt_set, PromptSet) else prompt_set
        return [self.process_output(p, o, **kwargs) for p, o in zip(prompts, output_set)]

class GeneratorBase:
    """Base class for prompt generators that can produce single or multiple prompts."""

    def generate_from_prompt(self, prompt_obj: Prompt, **kwargs) -> Union[Prompt, PromptSet]:
        """
        Override in subclasses to produce a single Prompt or a PromptSet.
        Default behavior returns the original Prompt unchanged.
        """
        return prompt_obj

    def generate_from_prompt_set(
        self,
        prompts: Union[List[Prompt], PromptSet],
        **kwargs
    ) -> PromptSet:
        """
        Dispatcher for multiple prompts. Calls `generate_from_prompt` for each item
        and wraps results into a PromptSet.
        """
        # Wrap list in PromptSet if needed
        if isinstance(prompts, list):
            prompts = PromptSet(prompts=prompts)

        result_set = PromptSet(output_type="multi")
        for p in prompts:
            res = self.generate_from_prompt(p, **kwargs)
            if isinstance(res, Prompt):
                result_set.add_prompt(res)
            elif isinstance(res, PromptSet):
                result_set.extend_prompts(list(res))

        # Preserve any tags from the original PromptSet
        if hasattr(prompts, "tags"):
            result_set.tags.update(prompts.tags)

        return result_set

class MutatorPlugin(BasePlugin):
    """Generators that specifically mutate prompts."""
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