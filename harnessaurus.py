import argparse
import os
import json
from datetime import datetime
from harness import run_batch_test, GPTModel, batchify
from plugin_loader import load_plugin
from result_aggregator import ResultAggregator


def load_list_from_file(path):
    if not path or not os.path.isfile(path):
        return []
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext == '.json':
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x.strip() for x in data if x.strip()]
            else:
                raise ValueError(f"{path} must be JSON array of strings")
        else:
            return [line.strip() for line in f if line.strip()]

def parse_plugins(plugin_specs):
    plugins = []
    for spec in plugin_specs:
        # spec can be "module.ClassName" or "module.ClassName:param=value,param2=val2"
        if ':' in spec:
            path, params_str = spec.split(':', 1)
            kwargs = {}
            for pair in params_str.split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    kwargs[k.strip()] = v.strip()
                else:
                    kwargs[pair.strip()] = True
            plugin = load_plugin(path, **kwargs)
        else:
            plugin = load_plugin(spec)
        plugins.append(plugin)
    return plugins


def main():
    parser = argparse.ArgumentParser(description="Red Team Test Harness")
    parser.add_argument('--prompts', required=True, help='File path for prompts (.txt or .json)')
    parser.add_argument('--forbidden_keywords', help='File path for forbidden keywords')
    parser.add_argument('--evasive_phrases', help='File path for evasive phrases')
    parser.add_argument('--plugins', nargs='+', required=False,
                        default=[
                            'zero_width_injector.ZeroWidthInjector',
                            'homoglyph_substitutor.HomoglyphSubstitutor',
                            'forbidden_keyword_detector.ForbiddenKeywordDetector',
                            'advanced_output_analyzer.AdvancedOutputAnalyzer',
                            'detoxify_plugin.DetoxifyPlugin',
                            'hidden_injection_detector.HiddenPromptInjectionDetector',
                            'json_logger.JsonLogger',
                        ],
                        help='List of plugins to load with optional params like mod.Class:param=val')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_workers', type=int, default=4)

    # Changed this from model_path to server_url to reflect llama-server usage
    parser.add_argument('--server_url', required=True, help='llama-server base URL, e.g. http://localhost:6589')
    parser.add_argument('--model_name', default='llama', help='Model name for llama-server API')

    args = parser.parse_args()

    # Load prompts as a list of strings
    prompts = load_list_from_file(args.prompts)

    # Pass file paths directly to plugins for internal loading
    forbidden_keywords_file = args.forbidden_keywords
    evasive_phrases_file = args.evasive_phrases

    plugin_param_map = {
        'forbidden_keyword_detector.ForbiddenKeywordDetector': {'keywords': forbidden_keywords_file},
        'advanced_output_analyzer.AdvancedOutputAnalyzer': {'evasive_phrases_file': evasive_phrases_file},
        'hidden_injection_detector.HiddenPromptInjectionDetector': {'homoglyph_file': 'data/homoglyphs.txt'},
    }

    plugins = []
    for spec in args.plugins:
        params = plugin_param_map.get(spec, {})
        plugin = load_plugin(spec, **params)
        plugins.append(plugin)

    # Initialize GPTModel with server URL and model name
    model = GPTModel(args.server_url, args.model_name)

    aggregator = ResultAggregator()

    batches = list(batchify(prompts, args.batch_size))

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_batch_test, batch, model, plugins, aggregator) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for r in results:
                    print(json.dumps(r, indent=2, ensure_ascii=False))
            except Exception as e:
                print(f"Error in batch: {e}")

    report_dir = 'reports'
    print(f"Creating reports directory at: {report_dir}")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    csv_filename = os.path.join(report_dir, f'redteam_results_{timestamp}.csv')
    json_filename = os.path.join(report_dir, f'redteam_results_{timestamp}.json')

    aggregator.save_csv(csv_filename)
    aggregator.save_json(json_filename)

    print("\n=== SUMMARY ===")
    print(json.dumps(aggregator.generate_summary(), indent=2))

if __name__ == '__main__':
    main()