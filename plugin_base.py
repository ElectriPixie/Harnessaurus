# plugin_base.py
from typing import List, Union
from data_structures import Prompt, PromptSet, Output, Record, OutputType

class PluginBase:
    """Minimal plugin functionality: shared hooks and metadata."""
    def on_log(self, record_obj: Record) -> None:
        """Optional hook called after a Record is created."""
        pass

class DetectorPlugin(PluginBase):
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

class GeneratorBase(PluginBase):
    """Base class for prompt generators that can produce single or multiple prompts."""

    def generate_from_prompt(self, prompt_obj: Prompt, **kwargs) -> Union[Prompt, PromptSet]:
        """Override in subclasses to produce a single Prompt or a PromptSet."""
        return prompt_obj

    def generate_from_prompt_set(
        self,
        prompts: Union[List[Prompt], PromptSet],
        **kwargs
    ) -> PromptSet:
        """Call `generate_from_prompt` for each item and wrap results into a PromptSet."""
        if isinstance(prompts, list):
            prompts = PromptSet(prompts=prompts)

        result_set = PromptSet(output_type="multi")
        for p in prompts:
            res = self.generate_from_prompt(p, **kwargs)
            if isinstance(res, Prompt):
                result_set.add_prompt(res)
            elif isinstance(res, PromptSet):
                result_set.extend_prompts(list(res))
            else:
                raise TypeError(f"Generator returned unexpected type: {type(res)}")

        if hasattr(prompts, "tags"):
            result_set.tags.update(prompts.tags)

        return result_set

    def _process_outputs(
        self,
        prompt_obj: Prompt,
        run_prompt: RunPrompt,
        model: GPTModel,
        aggregator: ResultAggregator,
        pm: PluginManager
    ) -> List[Record]:
        """
        Default processing: run inference, apply detectors, log results.
        Can be overridden by subclasses for custom behavior.
        """
        from data_structures import Record  # Only if needed for default implementation
        results: List[Record] = []

        # Default iterator logic
        iterator = 4 if getattr(prompt_obj, "has_context", False) else run_prompt.iterator

        # Run clean output first
        clean_output = run_model_inference(
            model,
            prompt_obj,
            iterator,
            run_prompt.max_tokens_per_chunk,
            run_prompt.max_iterations,
            run_prompt.flip_negate
        )

        # Run detectors
        clean_output = pm.process_output(
            prompt_obj=prompt_obj,
            output_obj=clean_output,
            plugins_to_apply=getattr(run_prompt, "use_detectors", [])
        )

        if not run_prompt.use_mutated:
            record = Record(
                original_prompt="\n".join(prompt_obj.prompt_list),
                mutated_prompt=None,
                clean_output=clean_output,
                mutated_output=None,
                mutation_iteration=0,
                run_dir=run_prompt.run_dir
            )
            aggregator.add_record(record)
            pm.process_log(record)
            results.append(record)
            return results

        # Mutated runs
        max_mutations = getattr(run_prompt, "max_mutations", 1)
        for mutation_index in range(1, max_mutations + 1):
            mutated_prompt = pm.process_prompt(
                prompt_obj=prompt_obj,
                plugins_to_apply=getattr(run_prompt, "use_mutators", [])
            )

            mutated_infer_output = run_model_inference(
                model,
                mutated_prompt,
                iterator,
                run_prompt.max_tokens_per_chunk,
                run_prompt.max_iterations,
                run_prompt.flip_negate
            )

            mutated_output = pm.process_output(
                prompt_obj=mutated_prompt,
                output_obj=mutated_infer_output,
                plugins_to_apply=getattr(run_prompt, "use_detectors", [])
            )

            record = Record(
                original_prompt="\n".join(prompt_obj.prompt_list),
                mutated_prompt="\n".join(mutated_prompt.prompt_list),
                clean_output=clean_output,
                mutated_output=mutated_output,
                mutation_iteration=mutation_index,
                run_dir=run_prompt.run_dir
            )
            aggregator.add_record(record)
            pm.process_log(record)
            results.append(record)

        return results

    def run_generated(
        self,
        generated: Union[Prompt, PromptSet],
        run_prompt: RunPrompt,
        model: GPTModel,
        aggregator: ResultAggregator,
        pm: PluginManager
    ) -> List[Record]:
        """Run the generated Prompt or PromptSet through the process_outputs hook."""
        if isinstance(generated, Prompt):
            prompts = [generated]
        elif isinstance(generated, PromptSet):
            prompts = list(generated)
        elif isinstance(generated, list) and all(isinstance(p, Prompt) for p in generated):
            prompts = generated
        else:
            raise TypeError(f"Unexpected type returned by generator: {type(generated)}")

        all_records = []
        for prompt_item in prompts:
            all_records.extend(self._process_outputs(prompt_item, run_prompt, model, aggregator, pm))
        return all_records

class MutatorPlugin(PluginBase):
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