from typing import List, Union, Callable
from data_structures import Prompt, PromptSet, Record, Output, RunPrompt
from runner_utils import run_model_inference
import copy

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
        return prompt_obj

    def generate_from_prompt_set(
        self,
        prompts: Union[List[Prompt], PromptSet],
        **kwargs
    ) -> PromptSet:
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
        run_prompt: "RunPrompt",
        model: "GPTModel",
        aggregator: "ResultAggregator",
        pm: "PluginManager",
    ) -> list[Record]:
        """
        Run inference on a prompt (including mutations), apply detectors, log results, and
        return a list of Record objects with all analysis included.
        """

        results: list[Record] = []
        iterator = 4 if getattr(prompt_obj, "has_context", False) else run_prompt.iterator

        def normalize_output(output_or_list):
            """Ensure we always have a list of Output objects with channels dict."""
            if isinstance(output_or_list, Output):
                if not hasattr(output_or_list, "channels") or output_or_list.channels is None:
                    output_or_list.channels = {}
                return [output_or_list]
            elif isinstance(output_or_list, list):
                for o in output_or_list:
                    if not hasattr(o, "channels") or o.channels is None:
                        o.channels = {}
                return output_or_list
            else:
                raise TypeError(f"Unexpected output type: {type(output_or_list)}")

        # --- Run clean outputs first ---
        clean_outputs = run_model_inference(
            model,
            prompt_obj,
            iterator,
            run_prompt.max_tokens_per_chunk,
            run_prompt.max_iterations,
            run_prompt.flip_negate,
            run_prompt.legacy_mode,
        )
        clean_outputs = normalize_output(clean_outputs)

        # Apply detectors
        clean_outputs = [
            pm.process_output(
                prompt_obj=prompt_obj,
                output_obj=o,
                plugins_to_apply=getattr(run_prompt, "use_detectors", [])
            )
            for o in clean_outputs
        ]

        # If mutated runs are disabled, store clean record and return
        if not getattr(run_prompt, "use_mutated", False):
            record = Record(
                original_prompt="\n".join([el["text"] for el in prompt_obj.prompt_list]),
                mutated_prompt=None,
                clean_outputs=clean_outputs,
                mutated_outputs=[],
                mutation_iteration=0,
                run_dir=run_prompt.run_dir,
            )
            aggregator.add_record(record)
            pm.process_log(record)
            results.append(record)
            return results

        # --- Handle mutated prompts ---
        max_mutations = getattr(run_prompt, "max_mutations", 1)
        for mutation_index in range(1, max_mutations + 1):
            mutated_prompt = pm.process_prompt(
                prompt_obj=copy.deepcopy(prompt_obj),
                plugins_to_apply=getattr(run_prompt, "use_mutators", [])
            )

            mutated_outputs = run_model_inference(
                model,
                mutated_prompt,
                iterator,
                run_prompt.max_tokens_per_chunk,
                run_prompt.max_iterations,
                run_prompt.flip_negate,
                run_prompt.legacy_mode,
            )
            mutated_outputs = normalize_output(mutated_outputs)

            # Apply detectors
            mutated_outputs = [
                pm.process_output(
                    prompt_obj=mutated_prompt,
                    output_obj=o,
                    plugins_to_apply=getattr(run_prompt, "use_detectors", [])
                )
                for o in mutated_outputs
            ]

            record = Record(
                original_prompt="\n".join([el["text"] for el in prompt_obj.prompt_list]),
                mutated_prompt="\n".join([el["text"] for el in mutated_prompt.prompt_list]),
                clean_outputs=clean_outputs,
                mutated_outputs=mutated_outputs,
                mutation_iteration=mutation_index,
                run_dir=run_prompt.run_dir,
            )
            aggregator.add_record(record)
            pm.process_log(record)
            results.append(record)

            # Stop early if RefusalDetector accepted and loop=False
            if not getattr(run_prompt, "loop", True):
                refusal_statuses = [
                    o.analysis.get("RefusalDetector", {}).get("status") for o in mutated_outputs
                ]
                if "accepted" in refusal_statuses:
                    print(f"[INFO] RefusalDetector accepted at mutation {mutation_index}, stopping early")
                    break

        return results

    def run_generated(
        self,
        generated: Union[Prompt, PromptSet],
        run_prompt: "RunPrompt",
        model: "GPTModel",
        aggregator: "ResultAggregator",
        pm: "PluginManager"
    ) -> List[Record]:
        """Run the generated Prompt or PromptSet through the _process_outputs hook."""

        # Normalize to flat list of Prompts
        if isinstance(generated, Prompt):
            prompts = [generated]
        elif isinstance(generated, PromptSet):
            prompts = list(generated)
        elif isinstance(generated, list) and all(isinstance(p, Prompt) for p in generated):
            prompts = generated
        else:
            raise TypeError(f"Unexpected type returned by generator: {type(generated)}")

        all_records = []
        for prompt_obj in prompts:
            all_records.extend(
                self._process_outputs(
                    prompt_obj=prompt_obj,
                    run_prompt=run_prompt,
                    model=model,
                    aggregator=aggregator,
                    pm=pm
                )
            )
        return all_records


class MutatorPlugin(PluginBase):
    """Plugins that mutate prompts."""

    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        return prompt_obj

    def process_prompt_set(
        self,
        prompt_set: Union[List[Prompt], PromptSet],
        **kwargs
    ) -> PromptSet:
        if isinstance(prompt_set, list):
            prompt_set = PromptSet(prompts=prompt_set)

        result_set = PromptSet(output_type="multi")
        for p in prompt_set:
            result_set.add_prompt(self.process_prompt(p, **kwargs))

        if hasattr(prompt_set, "tags"):
            result_set.tags.update(prompt_set.tags)
        return result_set