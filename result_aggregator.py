import csv
import json
import os
from collections import defaultdict
from typing import List, Dict, Any
from data_structures import Record

class ResultAggregator:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.total_prompts: int = 0
        self.plugin_flags: defaultdict[str, int] = defaultdict(int)
        self.plugin_suspicious: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.plugin_numeric_metrics: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.records: List[Record] = []

    def _dprint(self, *args):
        if self.debug:
            print("[ResultAggregator]", *args)

    def add_record(self, record: Record):
        """Process a Record object and update aggregator stats."""
        self.records.append(record)
        self.total_prompts += 1

        for output_attr in ['clean_output', 'mutated_output']:
            output: Record | None = getattr(record, output_attr)
            if output is None or not hasattr(output, 'analysis') or not isinstance(output.analysis, dict):
                continue

            analysis: Dict[str, Any] = output.analysis
            for plugin_name, plugin_result in analysis.items():
                if not isinstance(plugin_result, dict):
                    continue

                # Count flagged occurrences
                if plugin_result.get('flagged'):
                    self.plugin_flags[plugin_name] += 1

                # Count suspicious boolean flags
                for flag, val in plugin_result.items():
                    if isinstance(val, bool) and (flag.endswith('_suspected') or flag.endswith('_detected')) and val:
                        self.plugin_suspicious[(plugin_name, flag)] += 1

                # Collect numeric metrics
                for metric, val in plugin_result.items():
                    if isinstance(val, (int, float)):
                        self.plugin_numeric_metrics[plugin_name][metric].append(val)

    def generate_summary(self) -> Dict[str, Any]:
        total = max(self.total_prompts, 1)

        avg_metrics = {
            plugin: {metric: (sum(vals) / len(vals) if vals else None)
                     for metric, vals in metrics.items()}
            for plugin, metrics in self.plugin_numeric_metrics.items()
        }

        suspicious_counts = {f"{plugin}_{flag}": count for (plugin, flag), count in self.plugin_suspicious.items()}

        summary = {
            'total_prompts_tested': self.total_prompts,
            'plugin_flag_counts': dict(self.plugin_flags),
            'plugin_flag_percentages': {k: v / total * 100 for k, v in self.plugin_flags.items()},
            'plugin_suspicious_counts': suspicious_counts,
            'plugin_suspicious_percentages': {k: v / total * 100 for k, v in suspicious_counts.items()},
            'plugin_average_metrics': avg_metrics
        }

        return summary

    def save_json(self, filepath: str):
        """Save all records and summary to JSON, flush before return."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump({
                    'summary': self.generate_summary(),
                    'records': [r.to_dict() for r in self.records]
                }, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            self._dprint(f"Saved JSON: {filepath}")
        except Exception as e:
            print(f"[ResultAggregator] Failed to save JSON: {e}")

    def save_csv(self, filepath: str):
        """Save all records to CSV, flush before return."""
        if not self.records:
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        fieldnames = list(self.records[0].to_dict().keys())
        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.records:
                    writer.writerow(rec.to_dict())
                f.flush()
                os.fsync(f.fileno())
            self._dprint(f"Saved CSV: {filepath}")
        except Exception as e:
            print(f"[ResultAggregator] Failed to save CSV: {e}")
