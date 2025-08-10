import argparse
import os
import json
import csv
from datetime import datetime
from harness import run_batch_test, GPTModel, batchify
from plugin_loader import load_plugin
from result_aggregator import ResultAggregator
from critical_filter import CriticalRecordFilter
import concurrent.futures

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
                            'refusal_detector.RefusalDetector',
                            'reward_hacking_detector.RewardHackingDetector',
                            'deception_detector.DeceptionDetector',
                            'data_exfiltration_detector.DataExfiltrationDetector',
                            'harmful_tool_use_monitor.HarmfulToolUseMonitor',
                            'sabotage_detector.SabotageDetector',
                            'sandbagging_detector.SandbaggingDetector',
                            'hidden_motivation_detector.HiddenMotivationDetector',
                            'evaluation_awareness_detector_advanced.EvaluationAwarenessDetector',
                        ],
                        help='List of plugins to load with optional params like mod.Class:param=val')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_workers', type=int, default=4)
    parser.add_argument('--server_url', required=True, help='llama-server base URL, e.g. http://localhost:6589')
    parser.add_argument('--model_name', default='llama', help='Model name for llama-server API')

    args = parser.parse_args()

    # Load prompts as a list of strings
    prompts = load_list_from_file(args.prompts)

    # Timestamp for filenames
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Plugin parameter mappings
    plugin_param_map = {
        'forbidden_keyword_detector.ForbiddenKeywordDetector': {'keywords': args.forbidden_keywords},
        'advanced_output_analyzer.AdvancedOutputAnalyzer': {'evasive_phrases_file': args.evasive_phrases},
        'hidden_injection_detector.HiddenPromptInjectionDetector': {'homoglyph_file': 'data/homoglyphs.txt'},
    }

    # Load plugins with parameters
    plugins = []
    for spec in args.plugins:
        params = plugin_param_map.get(spec, {})
        if spec == 'json_logger.JsonLogger':
            params = dict(params)
            params['timestamp'] = timestamp
        plugin = load_plugin(spec, **params)
        plugins.append(plugin)

    # Initialize model, aggregator, and critical filter
    model = GPTModel(args.server_url, args.model_name)
    aggregator = ResultAggregator()
    critical_filter = CriticalRecordFilter()

    # Prepare report directory and files early
    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)

    csv_filename = os.path.join(report_dir, f'redteam_results_{timestamp}.csv')
    json_filename = os.path.join(report_dir, f'redteam_results_{timestamp}.jsonl')  # use jsonl for incremental writes

    # Open CSV and JSONL for progressive writing
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    json_file = open(json_filename, 'a', encoding='utf-8')

    # Define CSV columns - flat fields for main info, you can extend if you want
    csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff']
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Batch prompts
    batches = list(batchify(prompts, args.batch_size))

    # Run batches in ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_batch_test, batch, model, plugins, aggregator) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for r in results:
                    # Print nicely
                    print(json.dumps(r, indent=2, ensure_ascii=False))

                    # Write full logs
                    csv_writer.writerow({
                        'original_prompt': r['original_prompt'],
                        'mutated_prompt': r['mutated_prompt'],
                        'clean_output': r['clean_output'],
                        'mutated_output': r['mutated_output'],
                        'output_diff': r['output_diff'],
                    })
                    json_file.write(json.dumps(r) + '\n')

                    # Add to critical filter if critical
                    if critical_filter.is_critical(r):
                        critical_filter.add_record(r)

            except Exception as e:
                print(f"Error in batch: {e}")

    # Close full report files
    csv_file.close()
    json_file.close()

    # Save final aggregator full reports if needed
    aggregator.save_csv(csv_filename.replace('.csv', '_full.csv'))
    aggregator.save_json(json_filename.replace('.jsonl', '_full.json'))

    # Save critical filtered reports
    critical_csv_filename = os.path.join(report_dir, f'redteam_critical_{timestamp}.csv')
    critical_json_filename = os.path.join(report_dir, f'redteam_critical_{timestamp}.json')

    critical_filter.save_csv(critical_csv_filename)
    critical_filter.save_json(critical_json_filename)

    print("\n=== SUMMARY ===")
    print(json.dumps(aggregator.generate_summary(), indent=2))


if __name__ == '__main__':
    main()