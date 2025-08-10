import csv
import json
import os
from collections import defaultdict

class ResultAggregator:
    def __init__(self):
        self.total_prompts = 0
        self.plugin_flags = defaultdict(int)
        self.plugin_suspicious = defaultdict(int)  # {(plugin, suspicious_flag): count}
        self.plugin_numeric_metrics = defaultdict(lambda: defaultdict(list))  # {plugin: {metric: [values]}}
        self.records = []

    def add_record(self, record: dict):
        self.records.append(record)
        self.total_prompts += 1
        for key in ['analysis_clean', 'analysis_mutated']:
            analysis = record.get(key)
            if not analysis:
                continue  # skip if missing

            for plugin_name, plugin_result in analysis.items():
                if not isinstance(plugin_result, dict):
                    continue  # skip malformed plugin result

                # Count flagged occurrences safely
                if plugin_result.get('flagged'):
                    self.plugin_flags[plugin_name] += 1

                # Count suspicious boolean flags safely
                for suspicious_key in [
                    'hidden_motivation_suspected',
                    'evaluation_awareness_detected',
                    'sandbagging_suspected',
                    'reward_hacking_suspected',
                    'sabotage_suspected'
                ]:
                    if plugin_result.get(suspicious_key):
                        self.plugin_suspicious[(plugin_name, suspicious_key)] += 1

                # Collect numeric metrics if present and valid
                for metric in [
                    'score', 'lexical_diversity', 'average_word_length',
                    'length_change_from_last', 'semantic_similarity_to_last',
                    'hedging_phrases_found', 'vague_phrases_found', 'evasive_phrases_found',
                    'hedging_count', 'reasons_count'
                ]:
                    val = plugin_result.get(metric)
                    if isinstance(val, (int, float)):
                        self.plugin_numeric_metrics[plugin_name][metric].append(val)

    def generate_summary(self):
        total = self.total_prompts or 1

        avg_metrics = {}
        for plugin, metrics in self.plugin_numeric_metrics.items():
            avg_metrics[plugin] = {}
            for metric, values in metrics.items():
                if values:
                    avg_metrics[plugin][metric] = sum(values) / len(values)
                else:
                    avg_metrics[plugin][metric] = None

        suspicious_counts = {
            f"{plugin}_{key}": count
            for (plugin, key), count in self.plugin_suspicious.items()
        }

        return {
            'total_prompts_tested': self.total_prompts,
            'plugin_flag_counts': dict(self.plugin_flags),
            'plugin_suspicious_counts': suspicious_counts,
            'plugin_flag_percentages': {k: (v / total * 100) for k, v in self.plugin_flags.items()},
            'plugin_suspicious_percentages': {k: (v / total * 100) for k, v in suspicious_counts.items()},
            'plugin_average_metrics': avg_metrics,
        }

    def save_csv(self, filepath: str):
        keys = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output']
        plugin_keys = set()
        numeric_metrics = set()

        for record in self.records:
            for key in ['analysis_clean', 'analysis_mutated']:
                analysis = record.get(key)
                if not analysis:
                    continue
                plugin_keys.update(analysis.keys())
                for plugin_result in analysis.values():
                    if not isinstance(plugin_result, dict):
                        continue
                    numeric_metrics.update(
                        metric for metric, val in plugin_result.items()
                        if isinstance(val, (int, float))
                    )

        plugin_keys = sorted(plugin_keys)
        numeric_metrics = sorted(numeric_metrics)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = keys
            header += [f"{p}_clean_flagged" for p in plugin_keys]
            header += [f"{p}_mutated_flagged" for p in plugin_keys]

            for p in plugin_keys:
                for metric in numeric_metrics:
                    header.append(f"{p}_clean_{metric}")
            for p in plugin_keys:
                for metric in numeric_metrics:
                    header.append(f"{p}_mutated_{metric}")

            writer.writerow(header)

            for r in self.records:
                row = [r.get(k, '') for k in keys]

                for p in plugin_keys:
                    row.append(str(r.get('analysis_clean', {}).get(p, {}).get('flagged', False)))
                for p in plugin_keys:
                    row.append(str(r.get('analysis_mutated', {}).get(p, {}).get('flagged', False)))

                for p in plugin_keys:
                    for metric in numeric_metrics:
                        val = r.get('analysis_clean', {}).get(p, {}).get(metric, '')
                        if isinstance(val, float):
                            val = f"{val:.4f}"
                        row.append(val)
                for p in plugin_keys:
                    for metric in numeric_metrics:
                        val = r.get('analysis_mutated', {}).get(p, {}).get(metric, '')
                        if isinstance(val, float):
                            val = f"{val:.4f}"
                        row.append(val)

                writer.writerow(row)

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.generate_summary(),
                'records': self.records,
            }, f, indent=2)