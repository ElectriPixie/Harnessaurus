def analyze_plugin_results(plugin_outputs, score_func):
    """
    plugin_outputs: dict of {plugin_name: plugin_result}
    score_func: function that takes plugin_result and returns a score
    
    Returns: dict of {plugin_name: {'result': plugin_result, 'score': score}}
    """
    results_with_scores = {}

    for plugin_name, result in plugin_outputs.items():
        # Skip logger plugin
        if plugin_name.lower() == "logger":
            continue
        
        # Skip empty or falsey results
        if not result:
            continue
        
        # Apply score function
        score = score_func(result)
        
        results_with_scores[plugin_name] = {
            "result": result,
            "score": score
        }

    return results_with_scores
