from typing import Union
from plugin_base import GeneratorBase
from data_structures import Prompt, PromptSet

class PromptGenerator(GeneratorBase):
    """A trivial generator that returns the prompt as-is."""

    def generate_from_prompt(self, prompt_obj: Prompt, **kwargs) -> Union[Prompt, PromptSet]:
        # Simply return the original prompt without any changes
        return prompt_obj