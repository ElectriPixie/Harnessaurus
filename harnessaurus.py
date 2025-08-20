# main.py
import argparse
import os
import json
import csv
from datetime import datetime

from plugin_loader import load_plugin, load_detector, load_generator, load_logger, load_mutator
from result_aggregator import ResultAggregator
from critical_filter import CriticalRecordFilter
from harness import GPTModel, run_prompt_test
from data_structures import Prompt, RunPrompt, Output, Record

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def load_list_from_file(path):
    if not path or not os.path.isfile(path):
        return []
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            data = json.load(f)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x.strip() for x in data if x.strip()]
            else:
                raise ValueError(f"{path} must be a JSON array of strings")
        else:
            return [line.strip() for line in f if line.strip()]

def chunkify(prompt, max_tokens_per_chunk):
    yield prompt  # Currently placeholder, can be improved to split large prompts

def merge_chunks(chunk_records):
    return chunk_records[-1] if chunk_records else None

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Red Team Test Harness")
    parser.add_argument("--prompts", required=True, help="File path for prompts (.txt or .json)")
    parser.add_argument("--forbidden_keywords", help="File path for forbidden keywords")
    parser.add_argument("--evasive_phrases", help="File path for evasive phrases")
    # Detectors
    parser.add_argument('--detectors', nargs='+', required=False, default=[
        #'advanced_output_analyzer.AdvancedOutputAnalyzer',
        'detoxify_plugin.DetoxifyPlugin',
        'forbidden_keyword_detector.ForbiddenKeywordDetector',
        'hidden_injection_detector.HiddenPromptInjectionDetector',
        'refusal_detector.RefusalDetector',
        'reward_hacking_detector.RewardHackingDetector',
        'deception_detector.DeceptionDetector',
        'data_exfiltration_detector.DataExfiltrationDetector',
        'harmful_tool_use_monitor.HarmfulToolUseMonitor',
        'sabotage_detector.SabotageDetector',
        'sandbagging_detector.SandbaggingDetector',
        'hidden_motivation_detector.HiddenMotivationDetector',
        'evaluation_awareness_detector_advanced.EvaluationAwarenessDetector',
        #'eveluation_awareness_detector.EvaluationAwarenessDetector',
    ])

    # Mutators
    parser.add_argument('--mutators', nargs='+', required=False, default=[
        'banned_word_mutator.BannedWordMutator',
        'word_mutator.WordMutator',
        'homoglyph_substitutor.HomoglyphSubstitutor',
        'zero_width_injector.ZeroWidthInjector',
    ])

    # Context Generators
    parser.add_argument('--generators', nargs='+', required=False, default=[
        'rationalization_generator.RationalizationGenerator',
        'prompt_generator.PromptGenerator',
        # Add any context-generating plugins here
    ])

    # Loggers
    parser.add_argument('--loggers', nargs='+', required=False, default=[
        'json_logger.JsonLogger',
    ])
    parser.add_argument("--server_url", help="llama-server base URL", default="http://localhost:6589")
    parser.add_argument("--model_name", default="llama", help="Model name for llama-server API")
    parser.add_argument("--max_tokens_per_chunk", type=int, default=256)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--use_mutated", action="store_true", default=False)
    parser.add_argument("--max_mutations", type=int, default=1)
    parser.add_argument("--mutate_until_accepted", action="store_true", default=False)
    parser.add_argument("--use_mutators", nargs="+", required=False, default=[])
    parser.add_argument("--flip_negate", action="store_true", default=False)
    parser.add_argument("--skip_lines", type=int, default=0)
    parser.add_argument("--last_mutator", default=None)
    parser.add_argument("--single_pass", action="store_true")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--iterative_exploit", action="store_true")
    parser.add_argument("--homoglyph_replace_prob", type=float, default=1.0)
    parser.add_argument('--use_generator', type=str, default="PromptGenerator",
                    help="class name for generator to use")
    args = parser.parse_args()

    if args.single_pass:
        iterator = 1
    elif args.iterative:
        iterator = 2
    elif args.iterative_exploit:
        iterator = 3
    else:
        iterator = 1

    DEBUG = args.debug
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("reports", f"redteam_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    full_csv_path = os.path.join(run_dir, "redteam_results_full.csv")
    full_json_path = os.path.join(run_dir, "redteam_results_full.json")
    crit_csv_path = os.path.join(run_dir, "redteam_critical.csv")
    crit_json_path = os.path.join(run_dir, "redteam_critical.json")
    args_json_path = os.path.join(run_dir, "run_arguments.json")

    with open(args_json_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print(f"[Saved] Run arguments: {args_json_path}")

    detector_param_map = {
        "forbidden_keyword_detector.ForbiddenKeywordDetector": {"keywords": args.forbidden_keywords},
        "advanced_output_analyzer.AdvancedOutputAnalyzer": {"evasive_phrases_file": args.evasive_phrases},
        "hidden_injection_detector.HiddenPromptInjectionDetector": {"homoglyph_file": "data/homoglyphs/homoglyphs.txt"},
    }
    mutator_param_map = {
        "homoglyph_substitutor.HomoglyphSubstitutor": {
            "path": "data/homoglyphs/",
            "datasets": [("homoglyph_set", "homoglyphs.txt")],
            "replace_prob": args.homoglyph_replace_prob,
        },
    }
    logger_param_map = {}
    generator_param_map = {}

    # Load detectors
    detectors = []
    for spec in args.detectors:
        module_name, class_name = spec.rsplit('.', 1)
        params = detector_param_map.get(spec, {})
        detector = load_detector(module_name, class_name=class_name, **params)
        detectors.append(detector)

    # Load mutators
    mutators = []
    for spec in args.mutators:
        module_name, class_name = spec.rsplit('.', 1)
        params = mutator_param_map.get(spec, {})
        mutator = load_mutator(module_name, class_name=class_name, **params)
        mutators.append(mutator)

    # Load loggers
    loggers = []
    for spec in args.loggers:
        module_name, class_name = spec.rsplit('.', 1)
        params = logger_param_map.get(spec, {})
        if spec == "json_logger.JsonLogger":
            params = dict(params)
            params["timestamp"] = timestamp
        logger = load_logger(module_name, class_name=class_name, **params)
        loggers.append(logger)

    # Load generators
    generators = []
    for spec in args.generators:
        module_name, class_name = spec.rsplit('.', 1)
        params = generator_param_map.get(spec, {})
        generator = load_generator(module_name, class_name=class_name, **params)
        generators.append(generator)

    model = GPTModel(args.server_url, args.model_name)
    aggregator = ResultAggregator()
    critical_filter = CriticalRecordFilter()

    all_records: list[Record] = []

    full_csv_fields = ["original_prompt", "mutated_prompt", "clean_output", "mutated_output", "analysis_clean", "analysis_mutated", "mutation_iteration", "run_dir", ]
    crit_csv_fields = ["original_prompt", "mutated_prompt", "clean_output", "mutated_output", "critical_analysis", "analysis_clean", "analysis_mutated", "run_dir"]

    with open(full_csv_path, "w", newline="", encoding="utf-8") as f_csv, \
         open(full_json_path, "w", encoding="utf-8") as f_json, \
         open(crit_csv_path, "w", newline="", encoding="utf-8") as crit_csv, \
         open(crit_json_path, "w", encoding="utf-8") as crit_json:

        writer = csv.DictWriter(f_csv, fieldnames=full_csv_fields)
        writer.writeheader()
        crit_writer = csv.DictWriter(crit_csv, fieldnames=crit_csv_fields)
        crit_writer.writeheader()

        prompts = load_list_from_file(args.prompts)

        # Define colors near the top of main()
        RED = "\033[31m"
        BLUE = "\033[34m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        MAGENTA = "\033[35m"
        YELLOW = "\033[33m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"

        # Example usage when printing prompts
        for i, text in enumerate(prompts, 1):
            if args.skip_lines and i < args.skip_lines:
                continue
            print(f"{BOLD}{BLUE}[{GREEN}Processing{BLUE}] {YELLOW}{i}/{len(prompts)} {BLUE}Prompt: {GREEN}{text}{RESET}")

            chunk_records: list[Record] = []

            for chunk in chunkify(text, args.max_tokens_per_chunk):
                chunk_prompt = Prompt(prompt_list=[chunk])
                run_prompt_chunk = RunPrompt(
                    use_generator=args.use_generator,
                    prompt_obj=chunk_prompt,
                    iterator=iterator,
                    flip_negate=args.flip_negate,
                    max_tokens_per_chunk=args.max_tokens_per_chunk,
                    max_iterations=args.max_iterations,
                    loop=not args.mutate_until_accepted,
                    use_mutated=args.use_mutated,
                    max_mutations=args.max_mutations,
                    use_mutators=args.use_mutators,
                    run_dir=run_dir,
                    detector_plugins=detectors,
                    generator_plugins=generators,
                    logger_plugins=loggers,
                    mutator_plugins=mutators,   # loaded plugin instances
                )

                recs = run_prompt_test(run_prompt_chunk, model, aggregator)
                if not isinstance(recs, list) or not all(isinstance(r, Record) for r in recs):
                    raise TypeError(f"Expected list[Record] from run_prompt_test, got {type(recs)}")

                for rec in recs:
                    rec.run_dir = run_dir
                    chunk_records.append(rec)
                    all_records.append(rec)

                    # Save full results immediately
                    writer.writerow(rec.to_dict())
                    f_json.write(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2) + "\n")
                    f_json.flush()
                    os.fsync(f_json.fileno())

                    # Save critical records immediately
                    if critical_filter.is_critical(rec.to_dict()):
                        critical_filter.add_record(rec.to_dict())
                        enriched_rec = critical_filter.critical_records[-1]
                        enriched_rec["run_dir"] = rec.run_dir

                        critical_analysis = enriched_rec.get("critical_analysis", {})
                        analysis_clean = critical_analysis.get("analysis_clean", {})
                        analysis_mutated = critical_analysis.get("analysis_mutated", {})

                        row = {
                            **enriched_rec,
                            "critical_analysis": json.dumps(critical_analysis, ensure_ascii=False),
                            "analysis_clean": json.dumps(analysis_clean, ensure_ascii=False),
                            "analysis_mutated": json.dumps(analysis_mutated, ensure_ascii=False),
                        }

                        crit_writer.writerow({k: row.get(k, "") for k in crit_csv_fields})
                        crit_json.write(json.dumps(enriched_rec, ensure_ascii=False, indent=2) + "\n")
                        crit_json.flush()
                        os.fsync(crit_json.fileno())

    # Print final summary
    print("\n=== SUMMARY ===")
    print(json.dumps(aggregator.generate_summary(), indent=2))

if __name__ == "__main__":
    main()
