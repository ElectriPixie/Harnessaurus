import argparse
import os
import json
import csv
from datetime import datetime
from plugin_loader import load_plugin
from result_aggregator import ResultAggregator
from critical_filter import CriticalRecordFilter
from harness import GPTModel, run_prompt_test  # <-- import run_prompt_test here

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
        'RewardHackingDetector': [],
        'DeceptionDetector': [],
        'DataExfiltrationDetector': ['final'],
        'HarmfulToolUseMonitor': ['final'],
        'SabotageDetector': [],
        'SandbaggingDetector': [],
        'HiddenMotivationDetector': [],
        'EvaluationAwarenessDetector': [],
    }

    all_records = []


    for prompt in prompts:
        #print(f"[Processing] Prompt {i}/{len(prompts)}...")
        recs = run_prompt_test(
            prompt=prompt,  # pass a single prompt string directly
            plugins=plugins,
            model=model,
            aggregator=aggregator,
            channel_map=channel_map,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            max_iterations=args.max_iterations,
        )
        #print(json.dumps(recs, indent=2, ensure_ascii=False))
        all_records.extend(recs)


        for r in recs:
            if critical_filter.is_critical(r):
                critical_filter.add_record(r)

    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    full_csv_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.csv')
    full_json_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.jsonl')
    crit_csv_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.csv')
    crit_json_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.json')

    with open(full_csv_path, 'w', newline='', encoding='utf-8') as f_csv, \
         open(full_json_path, 'w', encoding='utf-8') as f_json:

        full_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff']
        writer = csv.DictWriter(f_csv, fieldnames=full_csv_fields)
        writer.writeheader()

        for rec in all_records:
            writer.writerow({
                'original_prompt': rec['original_prompt'],
                'mutated_prompt': rec['mutated_prompt'],
                'clean_output': rec['clean_output'],
                'mutated_output': rec['mutated_output'],
                'output_diff': rec['output_diff'],
            })
            f_json.write(json.dumps(rec, ensure_ascii=False) + '\n')

    if critical_filter.critical_records:
        with open(crit_csv_path, 'w', newline='', encoding='utf-8') as crit_csv, \
             open(crit_json_path, 'w', encoding='utf-8') as crit_json:

            crit_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'output_diff', 'critical_analysis']
            crit_writer = csv.DictWriter(crit_csv, fieldnames=crit_csv_fields)
            crit_writer.writeheader()

            for crit_rec in critical_filter.critical_records:
                row = crit_rec.copy()
                row['critical_analysis'] = json.dumps(row.get('critical_analysis', {}), ensure_ascii=False, indent=2)
                crit_writer.writerow(row)
                crit_json.write(json.dumps(crit_rec, ensure_ascii=False, indent=2) + '\n')

    print(f"[Saved] Full CSV: {full_csv_path}")
    print(f"[Saved] Full JSON: {full_json_path}")
    print(f"[Saved] Critical CSV: {crit_csv_path}")
    print(f"[Saved] Critical JSON: {crit_json_path}")

    print("\n=== SUMMARY ===")
    print(json.dumps(aggregator.generate_summary(), indent=2))


if __name__ == '__main__':
    main()
