from typing import Union
from plugin_base import GeneratorBase
from data_structures import Prompt, PromptSet, Record, OutputType

class ReplayGenerator(GeneratorBase):
    """
    Takes a list of clean prompts, optionally mutates them,
    and returns a PromptSet for execution.
    """

    def __init__(self, use_mutators: bool = True):
        super().__init__()
        self.use_mutators = use_mutators
        self.can_ret_list = True
        
    def _process_outputs(
        self,
        prompt_obj: Prompt,
        run_prompt: "RunPrompt",
        model: "GPTModel",
        aggregator: "ResultAggregator",
        pm: "PluginManager",
    ) -> list[Record]:
        """
        Process a Prompt where prompt_list contains a clean prompt and a mutated prompt.
        Runs inference for both, applies detectors, and builds a Record.
        """

        results: list[Record] = []

        # Extract clean and mutated text from prompt_list
        clean_text = None
        mutated_text = None
        for el in prompt_obj.prompt_list:
            if el.get("type") == "clean":
                clean_text = el["text"]
            elif el.get("type") == "mutated":
                mutated_text = el["text"]

        if clean_text is None:
            raise ValueError("No clean prompt found in prompt_list")
        if mutated_text is None:
            mutated_text = clean_text  # fallback: if no mutated line, just use clean

        # Build Prompt objects for inference
        clean_prompt = Prompt(prompt_list=[{"text": clean_text, "type": "clean", "mutate": False}])
        mutated_prompt = Prompt(prompt_list=[{"text": mutated_text, "type": "mutated", "mutate": False}])

        # --- Run clean output ---
        clean_output: Output = run_model_inference(
            model,
            clean_prompt,
            iterator=None,  # not needed
            max_tokens_per_chunk=run_prompt.max_tokens_per_chunk,
            max_iterations=run_prompt.max_iterations,
            flip_negate=run_prompt.flip_negate
        )

        clean_output = pm.process_output(
            prompt_obj=clean_prompt,
            output_obj=clean_output,
            plugins_to_apply=getattr(run_prompt, "use_detectors", [])
        )

        # --- Run mutated output ---
        mutated_infer_output: Output = run_model_inference(
            model,
            mutated_prompt,
            iterator=None,  # not needed
            max_tokens_per_chunk=run_prompt.max_tokens_per_chunk,
            max_iterations=run_prompt.max_iterations,
            flip_negate=run_prompt.flip_negate
        )

        mutated_output = pm.process_output(
            prompt_obj=mutated_prompt,
            output_obj=mutated_infer_output,
            plugins_to_apply=getattr(run_prompt, "use_detectors", [])
        )

        # --- Build Record ---
        record = Record(
            original_prompt=clean_text,
            mutated_prompt=mutated_text,
            clean_output=clean_output,
            mutated_output=mutated_output,
            mutation_iteration=1,
            run_dir=run_prompt.run_dir
        )

        aggregator.add_record(record)
        pm.process_log(record)
        results.append(record)

        return results

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