# result_aggregator.py
import csv
import json
import os
from collections import defaultdict

class ResultAggregator:
    def __init__(self):
        self.total_prompts = 0
        self.plugin_flags = defaultdict(int)
        self.records = []

    def add_record(self, record: dict):
        self.records.append(record)
        self.total_prompts += 1
        for key in ['analysis_clean', 'analysis_mutated']:
            analysis = record.get(key, {})
            for plugin_name, plugin_result in analysis.items():
                if plugin_result.get('flagged'):
                    self.plugin_flags[plugin_name] += 1

    def generate_summary(self):
        total = self.total_prompts or 1
        return {
            'total_prompts_tested': self.total_prompts,
            'plugin_flag_counts': dict(self.plugin_flags),
            'plugin_flag_percentages': {k: (v / total * 100) for k, v in self.plugin_flags.items()},
        }

    def save_csv(self, filepath: str):
        keys = ['original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output']
        plugin_keys = set()
        for record in self.records:
            for key in ['analysis_clean', 'analysis_mutated']:
                analysis = record.get(key, {})
                plugin_keys.update(analysis.keys())
        plugin_keys = sorted(plugin_keys)

        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = keys + [f"{p}_clean_flagged" for p in plugin_keys] + [f"{p}_mutated_flagged" for p in plugin_keys]
            writer.writerow(header)

            for r in self.records:
                row = [r.get(k, '') for k in keys]
                for p in plugin_keys:
                    row.append(str(r.get('analysis_clean', {}).get(p, {}).get('flagged', False)))
                for p in plugin_keys:
                    row.append(str(r.get('analysis_mutated', {}).get(p, {}).get('flagged', False)))
                writer.writerow(row)

    def save_json(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': self.generate_summary(),
                'records': self.records,
            }, f, indent=2)
