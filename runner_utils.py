# runner_utils.py
from data_structures import Prompt, Output, Record
from typing import Callable

def run_model_inference(
    model,
    prompt_obj: Prompt,
    iterator: int = 1,
    max_chunk_tokens: int = 256,
    max_iterations: int = 8,
    flip_negate: bool = False
) -> Output:
    """
    Dispatch prompt to the correct model method based on iterator.

    Returns an Output object regardless of iterator type.
    """
    if iterator == 1:
        # Single-pass returns Output directly
        return model.infer_single_pass(prompt=prompt_obj)
    elif iterator == 2:
        # Iterative returns str -> wrap into Output
        text = model.infer_iterative(
            prompt=prompt_obj,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate_flag=flip_negate
        )
        return Output(prompt=prompt_obj, raw_output=text)
    elif iterator == 3:
        text = model.infer_iterative_exploit(
            prompt=prompt_obj,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate_flag=flip_negate
        )
        return Output(prompt=prompt_obj, raw_output=text)
    elif iterator == 4:
        outputs = model.infer_iterative_with_prompt_list(
            prompts=prompt_obj,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate_flag=flip_negate
        )
        return outputs[-1]  # Return last Output in list
    else:
        raise ValueError(f"Unsupported iterator value: {iterator}")


def run_prompt_test(
    run_prompt,
    model,
    aggregator,
    pm,
    run_inference_func: Callable = run_model_inference
) -> list[Record]:
    """
    Run a Prompt object through GeneratorBase._process_outputs using injected inference.

    Args:
        run_prompt: RunPrompt object
        model: GPTModel instance
        aggregator: ResultAggregator instance
        pm: PluginManager instance
        run_inference_func: function that wraps model inference

    Returns:
        List of Record objects
    """
    # Use generator from RunPrompt
    generator_plugins = run_prompt.generator_plugins
    generator = next((g for g in generator_plugins if g.__class__.__name__ == run_prompt.use_generator), None)
    if generator is None:
        raise ValueError(f"Generator '{run_prompt.use_generator}' not found in plugins")

    # Generate prompts (can be Prompt or PromptSet)
    generated = generator.generate_from_prompt(run_prompt.prompt_obj)

    # Run _process_outputs for each prompt
    all_records = generator.run_generated(
        generated=generated,
        run_prompt=run_prompt,
        model=model,
        aggregator=aggregator,
        pm=pm
    )

    return all_records
