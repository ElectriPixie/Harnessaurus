import argparse
import os
import json
import csv
import time
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

    prompts = load_list_from_file(args.prompts)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    plugin_param_map = {
        'forbidden_keyword_detector.ForbiddenKeywordDetector': {'keywords': args.forbidden_keywords},
        'advanced_output_analyzer.AdvancedOutputAnalyzer': {'evasive_phrases_file': args.evasive_phrases},
        'hidden_injection_detector.HiddenPromptInjectionDetector': {'homoglyph_file': 'data/homoglyphs.txt'},
    }

    plugins = []
    for spec in args.plugins:
        params = plugin_param_map.get(spec, {})
        if spec == 'json_logger.JsonLogger':
            params = dict(params)
            params['timestamp'] = timestamp
        plugin = load_plugin(spec, **params)
        plugins.append(plugin)

    model = GPTModel(args.server_url, args.model_name)
    aggregator = ResultAggregator()
    critical_filter = CriticalRecordFilter()

    report_dir = 'reports'
    os.makedirs(report_dir, exist_ok=True)

    full_csv_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.csv')
    full_json_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.jsonl')
    crit_csv_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.csv')
    crit_json_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.json')

    full_csv_file = open(full_csv_path, 'w', newline='', encoding='utf-8')
    full_json_file = open(full_json_path, 'w', encoding='utf-8')
    crit_csv_file = open(crit_csv_path, 'w', newline='', encoding='utf-8')
    crit_json_file = open(crit_json_path, 'w', encoding='utf-8')

    full_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff']
    full_csv_writer = csv.DictWriter(full_csv_file, fieldnames=full_csv_fields)
    full_csv_writer.writeheader()

    crit_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff', 'critical_analysis']
    crit_csv_writer = csv.DictWriter(crit_csv_file, fieldnames=crit_csv_fields)
    crit_csv_writer.writeheader()

    batches = list(batchify(prompts, args.batch_size))

    # Hardcoded channel_map example — update or later make configurable
    channel_map = {
        # plugin class name (no module) : list of channels it should receive
        'ZeroWidthInjector': [],
        'HomoglyphSubstitutor': [],
        'ForbiddenKeywordDetector': ['final'],
        'AdvancedOutputAnalyzer': [],
        'DetoxifyPlugin': ['final'],
        'HiddenPromptInjectionDetector': ['final'],
        'JsonLogger': [],
        'RefusalDetector': ['final'],
        'RewardHackingDetector': [],
        'DeceptionDetector': [],
        'DataExfiltrationDetector': ['final'],
        'HarmfulToolUseMonitor': ['final'],
        'SabotageDetector': [],
        'SandbaggingDetector': [],
        'HiddenMotivationDetector': [],
        'EvaluationAwarenessDetector': [],
        # Default fallback will pass full raw output if no key found
    }

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_batch_test, batch, model, plugins, aggregator, channel_map=channel_map)
            for batch in batches
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for r in results:
                    # Debug prints
                    print(f"[Main] Plugins clean: {list(r.get('analysis_clean', {}).keys())}")
                    print(f"[Main] Plugins mutated: {list(r.get('analysis_mutated', {}).keys())}")
                    print(json.dumps(r, indent=2, ensure_ascii=False))

                    # Write full report records incrementally
                    full_csv_writer.writerow({
                        'original_prompt': r.get('original_prompt', ''),
                        'mutated_prompt': r.get('mutated_prompt', ''),
                        'clean_output': r.get('clean_output', ''),
                        'mutated_output': r.get('mutated_output', ''),
                        'output_diff': r.get('output_diff', ''),
                    })
                    full_json_file.write(json.dumps(r, ensure_ascii=False) + '\n')

                    full_csv_file.flush()
                    full_json_file.flush()

                    aggregator.add_record(r)

                    if critical_filter.is_critical(r):
                        critical_filter.add_record(r)
                        last_crit = critical_filter.critical_records[-1]
                        crit_row = last_crit.copy()
                        crit_row['critical_analysis'] = json.dumps(crit_row['critical_analysis'], ensure_ascii=False, indent=2)
                        crit_csv_writer.writerow(crit_row)
                        crit_json_file.write(json.dumps(last_crit, ensure_ascii=False, indent=2) + '\n')

                        crit_csv_file.flush()
                        crit_json_file.flush()

            except Exception as e:
                print(f"Error in batch: {e}")

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds\n")

    full_csv_file.close()
    full_json_file.close()
    crit_csv_file.close()
    crit_json_file.close()

    print(f"[Saved] Full CSV: {full_csv_path}")
    print(f"[Saved] Full JSON: {full_json_path}")
    print(f"[Saved] Critical CSV: {crit_csv_path}")
    print(f"[Saved] Critical JSON: {crit_json_path}")

    print("=== SUMMARY ===")
    print(json.dumps(aggregator.generate_summary(), indent=2))


if __name__ == '__main__':
    main()