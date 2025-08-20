# main.py
import argparse
import os
import json
import csv
from datetime import datetime

from plugin_loader import load_plugin, load_detector, load_generator, load_logger, load_mutator
from result_aggregator import ResultAggregator
from critical_filter import CriticalRecordFilter
from data_structures import Prompt, PromptSet, RunPrompt, Output, Record
from gpt_model import GPTModel
from utils import debug_print
from runner_utils import run_prompt_test, run_model_inference
from plugin_manager import PluginManager
from prompt_processor import BasePromptProcessor, ReplayPromptProcessor
from typing import Type

DEBUG = False


PROCESSOR_MAP = {
    "base": BasePromptProcessor,
    "replay": ReplayPromptProcessor,
}

def make_processor(name: str, path: str) -> BasePromptProcessor:
    cls: Type[BasePromptProcessor] = PROCESSOR_MAP.get(name.lower())
    if not cls:
        raise ValueError(f"Unknown processor '{name}', valid options: {list(PROCESSOR_MAP.keys())}")
    return cls(path)

def chunkify(prompt_text: str, max_tokens_per_chunk: int):
    """
    Yield dicts directly compatible with Prompt.prompt_list.
    """
    # Just yield the single chunk directly
    yield {
        "text": prompt_text,
        "type": "prompt",
        "mutate": True
    }

def merge_chunks(chunk_records):
    return chunk_records[-1] if chunk_records else None

def run_prompt_test_wrapped(run_prompt: "RunPrompt", model: "GPTModel", aggregator: "ResultAggregator") -> list[Record]:
    pm = PluginManager(
        mutators=run_prompt.mutator_plugins,
        detectors=run_prompt.detector_plugins,
        loggers=run_prompt.logger_plugins
    )

    # Find generator plugin by name
    generator = next(
        (g for g in run_prompt.generator_plugins if g.__class__.__name__ == run_prompt.use_generator),
        None
    )
    if generator is None:
        raise ValueError(f"Generator '{run_prompt.use_generator}' not found")

    # Generate Prompt or PromptSet
    generated = generator.generate_from_prompt(run_prompt.prompt_obj)

    all_records: list[Record] = generator.run_generated(
        generated=generated,
        run_prompt=run_prompt,
        model=model,
        aggregator=aggregator,
        pm=pm
    )

    return all_records

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Red Team Test Harness")
    parser.add_argument("--prompts", required=True, help="File path for prompts (.txt or .json)")
    parser.add_argument("--forbidden_keywords", help="File path for forbidden keywords")
    parser.add_argument("--evasive_phrases", help="File path for evasive phrases")
    parser.add_argument('--detectors', nargs='+', required=False, default=[
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
    ])
    parser.add_argument('--mutators', nargs='+', required=False, default=[
        'banned_word_mutator.BannedWordMutator',
        'word_mutator.WordMutator',
        'homoglyph_substitutor.HomoglyphSubstitutor',
        'zero_width_injector.ZeroWidthInjector',
    ])
    parser.add_argument('--generators', nargs='+', required=False, default=[
        'replay_generator.ReplayGenerator',
        'rationalization_generator.RationalizationGenerator',
        'prompt_generator.PromptGenerator',
    ])
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
    parser.add_argument("--single_pass", action="store_true")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--iterative_exploit", action="store_true")
    parser.add_argument("--homoglyph_replace_prob", type=float, default=1.0)
    parser.add_argument("--word_replace_prob", type=float, default=1.0)
    parser.add_argument('--use_generator', type=str, default="PromptGenerator")
    parser.add_argument(
        "--processor",
        choices=["base", "replay"],  # add more as needed
        default="base",
        help="Which prompt processor to use"
    )
    args = parser.parse_args()

    # Determine iteration mode
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

    # Parameter maps
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
        "word_mutator.WordMutator": {
            "replace_prob": args.word_replace_prob
        }
    }
    logger_param_map = {}
    generator_param_map = {}

    # Load plugins
    detectors = [load_detector(*spec.rsplit('.', 1), **detector_param_map.get(spec, {})) for spec in args.detectors]
    mutators = [load_mutator(*spec.rsplit('.', 1), **mutator_param_map.get(spec, {})) for spec in args.mutators]
    loggers = []
    for spec in args.loggers:
        module_name, class_name = spec.rsplit('.', 1)
        params = logger_param_map.get(spec, {})
        if spec == "json_logger.JsonLogger":
            params = dict(params)
            params["timestamp"] = timestamp
        loggers.append(load_logger(module_name, class_name=class_name, **params))
    generators = [load_generator(*spec.rsplit('.', 1), **generator_param_map.get(spec, {})) for spec in args.generators]

    # Initialize model, aggregator, filters
    model = GPTModel(args.server_url, args.model_name)
    aggregator = ResultAggregator()
    critical_filter = CriticalRecordFilter()

    all_records: list[Record] = []

    full_csv_fields = ["original_prompt", "mutated_prompt", "clean_output", "mutated_output",
                       "analysis_clean", "analysis_mutated", "mutation_iteration", "run_dir"]
    crit_csv_fields = ["original_prompt", "mutated_prompt", "clean_output", "mutated_output",
                       "critical_analysis", "analysis_clean", "analysis_mutated", "run_dir"]

    # ANSI colors
    RED = "\033[31m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    with open(full_csv_path, "w", newline="", encoding="utf-8") as f_csv, \
         open(full_json_path, "w", encoding="utf-8") as f_json, \
         open(crit_csv_path, "w", newline="", encoding="utf-8") as crit_csv, \
         open(crit_json_path, "w", encoding="utf-8") as crit_json:

        writer = csv.DictWriter(f_csv, fieldnames=full_csv_fields)
        writer.writeheader()
        crit_writer = csv.DictWriter(crit_csv, fieldnames=crit_csv_fields)
        crit_writer.writeheader()

        processor = make_processor(args.processor, args.prompts)

        for i, prompt_list in enumerate(processor(), 1):
            if args.skip_lines and i < args.skip_lines:
                continue

            # Preview text for logging
            text_preview = ", ".join([p["text"] for p in prompt_list])
            print(f"{BOLD}{BLUE}[{GREEN}Processing{BLUE}] {YELLOW}{i}/{int(len(processor.prompts)/processor.diviser)} {BLUE}Prompt: {GREEN}{text_preview}{RESET}")

            chunk_records: list[Record] = []

            # Wrap prompt_list in Prompt object directly
            chunk_prompt = Prompt(prompt_list=prompt_list)
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
                mutator_plugins=mutators,
            )

            recs = run_prompt_test_wrapped(run_prompt_chunk, model, aggregator)

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

    print(f"\n{BOLD}{CYAN}=== SUMMARY ==={RESET}")
    print(json.dumps(aggregator.generate_summary(), indent=2))

if __name__ == "__main__":
    main()
