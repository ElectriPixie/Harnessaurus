#runner_utils.py
from data_structures import Prompt, Output, Record

def run_model_inference(
    model,
    prompt_obj,
    iterator=1,
    max_chunk_tokens=256,
    max_iterations=8,
    flip_negate=False
):
    """
    Dispatch prompt to the correct model method based on iterator.

    Args:
        model: the model object containing the inference methods
        prompt: a Prompt object or string
        iterator: chooses the method (1-4)
        max_chunk_tokens: max tokens for chunked methods
        max_iterations: max iterations for iterative methods
        flip_negate: whether to apply flip_negate in iterative methods

    Returns:
        The model Output
    """
    if iterator == 1:
        return model.infer_single_pass(prompt=prompt_obj)
    elif iterator == 2:
        return model.infer_iterative(
            prompt=prompt_obj,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate_flag=flip_negate
        )
    elif iterator == 3:
        return model.infer_iterative_exploit(
            prompt=prompt_obj,
            max_chunk_tokens=max_chunk_tokens,
            max_iterations=max_iterations,
            flip_negate_flag=flip_negate
        )
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
    run_inference_func  # injected inference function
) -> list[Record]:
    """Run a Prompt object through GeneratorBase._process_outputs."""
    generated = run_prompt.prompt_obj
    generator = run_prompt.generator_plugins
    return generator._process_outputs(
        prompt_obj=generated,
        run_prompt=run_prompt,
        aggregator=aggregator,
        pm=pm,
        run_inference_func=run_inference_func
    )
