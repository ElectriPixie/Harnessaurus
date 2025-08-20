from typing import Union
from plugin_base import GeneratorBase
from data_structures import Prompt, PromptSet, OutputType

class ReplayGenerator(GeneratorBase):
    """
    Takes a list of clean prompts, optionally mutates them,
    and returns a PromptSet for execution.
    """

    def __init__(self, use_mutators: bool = True):
        super().__init__()
        self.use_mutators = use_mutators
        self.can_ret_list = True

    def generate_from_prompt(
        self,
        prompt_obj: Prompt,
        mutation_index: int = 0,
        with_context: bool = True,
        full_set: bool = True
    ) -> Union[Prompt, PromptSet]:

        if not isinstance(prompt_obj, Prompt):
            raise TypeError("ReplayGenerator expects a Prompt object")

        prompt_set = PromptSet(output_type="multi")

        for base_text in prompt_obj.prompt_list:
            # Apply mutation if enabled
            mutated_text = self.mutate_prompt(base_text) if self.use_mutators else base_text

            prompt_set.add_prompt(Prompt(
                prompt_list=[mutated_text],
                has_context=with_context,
                tags={**prompt_obj.tags, "replay": True}
            ))

        return prompt_set

    def mutate_prompt(self, text: str) -> str:
        # Example mutation; replace this with your actual mutator logic
        return text + " [mutated]"