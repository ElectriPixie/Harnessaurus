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

# Placeholder: yields whole prompt as one chunk
def chunkify(prompt, max_tokens_per_chunk):
    yield prompt

# Placeholder: just returns last chunk record
def merge_chunks(chunk_records):
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
    parser.add_argument('--server_url', help='llama-server base URL, e.g. http://localhost:6589', default="http://localhost:6589")
    parser.add_argument('--model_name', default='llama', help='Model name for llama-server API')
    parser.add_argument('--max_tokens_per_chunk', type=int, default=256)
    parser.add_argument('--max_iterations', type=int, default=10)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--use_mutated', action='store_true', help='Enable mutated prompts and outputs', default=False)
    parser.add_argument('--max_mutations', type=int, default=1)
    parser.add_argument('--flip_negate', action='store_true', help='Enable flip negating', default=False)
    parser.add_argument('--iterator', type=int, default=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--single_pass",
        action="store_true",
        help="Run the harness in single-pass mode. Each prompt is processed once; no iterative mutation is applied."
    )
    group.add_argument(
        "--iterative",
        action="store_true",
        help="Run the harness in iterative mode. Prompts are chunked iteratively to test model vulnerabilities."
    )
    group.add_argument(
        "--iterative_exploit",
        action="store_true",
        help="Run the harness in iterative exploit mode. Prompts are chunked iteratively to test model vulnerabilities. Warning: this mode may be unstable."
    )

    args = parser.parse_args()
    if args.single_pass:
        iterator = 1
        print("Running in single-pass mode...")
    elif args.iterative:
        print(f"Running in iterative mode for up to {args.max_iterations} iterations...")
        iterator = 2
    elif args.iterative_exploit:
        print(f"Running in iterative exploit mode for up to {args.max_iterations} iterations...")
        iterator = 3
    else:
        iterator = 1
        print("No mode selected; defaulting to single-pass.")
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
    full_json_path = os.path.join(report_dir, f'redteam_results_{timestamp}_full.json')
    crit_csv_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.csv')
    crit_json_path = os.path.join(report_dir, f'redteam_critical_{timestamp}.json')

    full_csv_fields = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output', 'mutation_iteration']
    crit_csv_fields = [
        'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
        'critical_analysis', 'analysis_clean', 'analysis_mutated'
    ]

    with open(full_csv_path, 'w', newline='', encoding='utf-8') as f_csv, \
         open(full_json_path, 'w', encoding='utf-8') as f_json, \
         open(crit_csv_path, 'w', newline='', encoding='utf-8') as crit_csv, \
         open(crit_json_path, 'w', encoding='utf-8') as crit_json:

        writer = csv.DictWriter(f_csv, fieldnames=full_csv_fields)
        writer.writeheader()

        crit_writer = csv.DictWriter(crit_csv, fieldnames=crit_csv_fields)
        crit_writer.writeheader()

        for i, prompt in enumerate(prompts, 1):
            print(f"[Processing] Prompt {i}/{len(prompts)}...")

            chunk_records = []

            for chunk in chunkify(prompt, args.max_tokens_per_chunk):
                rec_list = run_prompt_test(
                    chunk,
                    model,
                    plugins,
                    aggregator,
                    channel_map=channel_map,
                    max_tokens_per_chunk=args.max_tokens_per_chunk,
                    max_iterations=args.max_iterations,
                    include_mutated_output=args.use_mutated,
                    max_mutations=args.max_mutations,
                    flip_negate=args.flip_negate,
                    iterator=iterator
                )

                if not isinstance(rec_list, list):
                    raise TypeError(f"Unexpected return type from run_prompt_test: {type(rec_list)}")

                for rec in rec_list:
                    if not isinstance(rec, dict):
                        debug_print(f"[Main] Skipping non-dict record: {type(rec)}")
                        continue

                    chunk_records.append(rec)
                    all_records.append(rec)

                    try:
                        writer.writerow({
                            'original_prompt': rec.get('original_prompt', ''),
                            'mutated_prompt': rec.get('mutated_prompt', ''),
                            'clean_output': rec.get('clean_output', ''),
                            'mutated_output': rec.get('mutated_output', ''),
                            'mutation_iteration': rec.get('mutation_iteration', ''),
                        })
                    except Exception as e:
                        debug_print(f"[Main] Failed writing full CSV row: {e}")

                    try:
                        f_json.write(json.dumps(rec, ensure_ascii=False, indent=2) + '\n')
                    except Exception as e:
                        debug_print(f"[Main] Failed writing full JSON line: {e}")

                    try:
                        if critical_filter.is_critical(rec):
                            critical_filter.add_record(rec)
                            enriched_rec = critical_filter.critical_records[-1]

                            critical_analysis = enriched_rec.get('critical_analysis', {})
                            analysis_clean = critical_analysis.get('analysis_clean', {})
                            analysis_mutated = critical_analysis.get('analysis_mutated', {})

                            row = enriched_rec.copy()
                            try:
                                row['critical_analysis'] = json.dumps(critical_analysis, ensure_ascii=False, indent=2)
                            except Exception:
                                row['critical_analysis'] = json.dumps(critical_analysis, ensure_ascii=False)

                            try:
                                row['analysis_clean'] = json.dumps(analysis_clean, ensure_ascii=False)
                            except Exception:
                                row['analysis_clean'] = ''

                            try:
                                row['analysis_mutated'] = json.dumps(analysis_mutated, ensure_ascii=False)
                            except Exception:
                                row['analysis_mutated'] = ''

                            filtered_row = {k: row.get(k, '') for k in crit_csv_fields}
                            try:
                                crit_writer.writerow(filtered_row)
                            except Exception as e:
                                debug_print(f"[Main] Failed writing critical CSV row: {e}")

                            try:
                                crit_json.write(json.dumps(enriched_rec, ensure_ascii=False, indent=2) + '\n')
                            except Exception as e:
                                debug_print(f"[Main] Failed writing critical JSON entry: {e}")

                    except Exception as e:
                        print(f"[Warning] critical_filter failed for a record: {e}")

        print(f"[Saved] Full CSV: {full_csv_path}")
        print(f"[Saved] Full JSON: {full_json_path}")
        print(f"[Saved] Critical CSV: {crit_csv_path}")
        print(f"[Saved] Critical JSON: {crit_json_path}")

        print("\n=== SUMMARY ===")
        print(json.dumps(aggregator.generate_summary(), indent=2))

if __name__ == '__main__':
    main()
