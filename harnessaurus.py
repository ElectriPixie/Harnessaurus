import argparse
import os
import json
import csv
from datetime import datetime
from plugin_loader import load_plugin
from result_aggregator import ResultAggregator
from critical_filter import CriticalRecordFilter
from harness import GPTModel, run_prompt_test

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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
                raise ValueError(f"{path} must be a JSON array of strings")
        else:
            return [line.strip() for line in f if line.strip()]

# You need to define how you want to chunk prompts.
# This is a simple placeholder that yields the entire prompt as one chunk.
def chunkify(prompt, max_tokens_per_chunk):
    # TODO: Replace with real chunking logic if needed
    yield prompt

# You need to define how to merge chunk records into one record.
# This is a placeholder that just returns the last chunk record.
def merge_chunks(chunk_records):
    # TODO: Replace with logic that merges chunk records meaningfully
    return chunk_records[-1] if chunk_records else {}

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Red Team Test Harness")
    parser.add_argument('--prompts', required=True, help='File path for prompts (.txt or .json)')
    parser.add_argument('--forbidden_keywords', help='File path for forbidden keywords')
    parser.add_argument('--evasive_phrases', help='File path for evasive phrases')
    parser.add_argument('--plugins', nargs='+', required=False,
                        default=[
                            'zero_width_injector.ZeroWidthInjector',
                            'homoglyph_substitutor.HomoglyphSubstitutor',
                            'forbidden_keyword_detector.ForbiddenKeywordDetector',
                            #'advanced_output_analyzer.AdvancedOutputAnalyzer',
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
    parser.add_argument('--server_url', required=True, help='llama-server base URL, e.g. http://localhost:6589')
    parser.add_argument('--model_name', default='llama', help='Model name for llama-server API')
    parser.add_argument('--max_tokens_per_chunk', type=int, default=256)
    parser.add_argument('--max_iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    args = parser.parse_args()
    DEBUG = args.debug

    debug_print("[Main] Starting with debug ON")

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

    channel_map = {
        'ZeroWidthInjector': [],
        'HomoglyphSubstitutor': [],
        'ForbiddenKeywordDetector': ['final'],
        'AdvancedOutputAnalyzer': [],
        'DetoxifyPlugin': ['final'],
        'HiddenPromptInjectionDetector': ['final'],
        'JsonLogger': [],
        'RefusalDetector': ['final'],
        'RewardHackingDetector': ['final'],
        'DeceptionDetector': ['final'],
        'DataExfiltrationDetector': ['final'],
        'HarmfulToolUseMonitor': ['final'],
        'SabotageDetector': ['final'],
        'SandbaggingDetector': [],
        'HiddenMotivationDetector': [],
        'EvaluationAwarenessDetector': [],
    }

    all_records = []

    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    full_csv_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.csv')
    full_json_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.jsonl')
    crit_csv_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.csv')
    crit_json_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.json')

    with open(full_csv_path, 'w', newline='', encoding='utf-8') as f_csv, \
         open(full_json_path, 'w', encoding='utf-8') as f_json, \
         open(crit_csv_path, 'w', newline='', encoding='utf-8') as crit_csv, \
         open(crit_json_path, 'w', encoding='utf-8') as crit_json:

        full_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff']
        writer = csv.DictWriter(f_csv, fieldnames=full_csv_fields)
        writer.writeheader()

        crit_csv_fields = [
            'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
            'output_diff', 'critical_analysis', 'analysis_clean', 'analysis_mutated'
        ]
        crit_writer = csv.DictWriter(crit_csv, fieldnames=crit_csv_fields)
        crit_writer.writeheader()

        for i, prompt in enumerate(prompts, 1):
            print(f"[Processing] Prompt {i}/{len(prompts)}...")

            chunk_records = []

            for chunk in chunkify(prompt, args.max_tokens_per_chunk):
                rec = run_prompt_test(
                    chunk,
                    model,
                    plugins,
                    aggregator,
                    channel_map=channel_map,
                    max_tokens_per_chunk=args.max_tokens_per_chunk,
                    max_iterations=args.max_iterations,
                )
                chunk_records.append(rec)
                all_records.append(rec)

            final_rec = merge_chunks(chunk_records)
            if critical_filter.is_critical(final_rec):
                critical_filter.add_record(final_rec)

            writer.writerow({
                'original_prompt': final_rec.get('original_prompt', ''),
                'mutated_prompt': final_rec.get('mutated_prompt', ''),
                'clean_output': final_rec.get('clean_output', ''),
                'mutated_output': final_rec.get('mutated_output', ''),
                'output_diff': final_rec.get('output_diff', ''),
            })
            f_json.write(json.dumps(final_rec, ensure_ascii=False) + '\n')

            if any(critical_filter.is_critical(r) for r in chunk_records):
                crit_rec = critical_filter.critical_records[-1]  # last added
                row = crit_rec.copy()
                row['critical_analysis'] = json.dumps(row.get('critical_analysis', {}), ensure_ascii=False, indent=2)
                row['analysis_clean'] = json.dumps(row.get('analysis_clean', {}), ensure_ascii=False)
                row['analysis_mutated'] = json.dumps(row.get('analysis_mutated', {}), ensure_ascii=False)
                filtered_row = {k: row.get(k, '') for k in crit_csv_fields}
                crit_writer.writerow(filtered_row)
                crit_json.write(json.dumps(crit_rec, ensure_ascii=False, indent=2) + '\n')

        print(f"[Saved] Full CSV: {full_csv_path}")
        print(f"[Saved] Full JSON: {full_json_path}")
        print(f"[Saved] Critical CSV: {crit_csv_path}")
        print(f"[Saved] Critical JSON: {crit_json_path}")

        print("\n=== SUMMARY ===")
        print(json.dumps(aggregator.generate_summary(), indent=2))


if __name__ == '__main__':
    main()
