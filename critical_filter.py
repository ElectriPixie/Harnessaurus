import csv
import json
import os
from typing import Optional, Dict, Any, List


class CriticalRecordFilter:
    """
    Filter records to identify 'critical' plugin flags indicating suspicious or problematic outputs.
    Supports:
    - Boolean flags based on key suffixes (e.g. 'flagged', 'detected', etc.)
    - Numeric thresholds on keys containing substrings like 'score', 'count', etc.
    - Special handling for refusal_detector plugin (always included if non-empty)

    Keeps a list of critical records with enriched critical_analysis data.
    File writing is done explicitly via save_csv / save_json methods.
    """

    CRITICAL_BOOL_SUFFIXES = [
        "flagged", "suspected", "detected", "error",
        "alert", "warning", "shift", "patterns",
    ]

    CRITICAL_NUMERIC_SUBSTRINGS = [
        "score", "count", "level", "severity",
    ]

    DEFAULT_NUMERIC_THRESHOLD = 5
    REFUSAL_DETECTOR_PLUGIN = "refusal_detector.RefusalDetector"

    def __init__(self, plugin_thresholds: Optional[Dict[str, float]] = None):
        self.plugin_thresholds = plugin_thresholds or {}
        self.critical_records: List[Dict[str, Any]] = []

    def _is_critical_boolean(self, key: str, value: Any) -> bool:
        return isinstance(value, bool) and any(key.lower().endswith(suffix) for suffix in self.CRITICAL_BOOL_SUFFIXES) and value

    def _is_critical_numeric(self, plugin_name: str, key: str, value: Any) -> bool:
        if isinstance(value, (int, float)) and any(substr in key.lower() for substr in self.CRITICAL_NUMERIC_SUBSTRINGS):
            threshold = self.plugin_thresholds.get(plugin_name, self.DEFAULT_NUMERIC_THRESHOLD)
            return value >= threshold
        return False

    def filter_record(self, record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        filtered = {"analysis_clean": {}, "analysis_mutated": {}}

        for analysis_key in filtered.keys():
            analysis = record.get(analysis_key, {})
            filtered[analysis_key] = {}

            for plugin_name, plugin_data in analysis.items():
                # Wrap boolean data for uniformity
                if isinstance(plugin_data, bool):
                    plugin_data = {"flagged": plugin_data}

                # Special case: refusal_detector
                if plugin_name == self.REFUSAL_DETECTOR_PLUGIN:
                    if isinstance(plugin_data, dict) and any(plugin_data.values()):
                        filtered[analysis_key][plugin_name] = plugin_data
                    continue

                if not isinstance(plugin_data, dict):
                    continue

                suspicious = {}
                for key, value in plugin_data.items():
                    if self._is_critical_boolean(key, value) or self._is_critical_numeric(plugin_name, key, value):
                        suspicious[key] = value

                if suspicious:
                    filtered[analysis_key][plugin_name] = suspicious

        return filtered

    def is_critical(self, record: Dict[str, Any]) -> bool:
        filtered = self.filter_record(record)
        return bool(filtered["analysis_clean"] or filtered["analysis_mutated"])

    def add_record(self, record: Dict[str, Any]):
        """
        Add record to memory if critical.
        Does NOT perform file writes.
        """
        filtered = self.filter_record(record)
        if filtered["analysis_clean"] or filtered["analysis_mutated"]:
            enriched_record = dict(record)
            enriched_record["critical_analysis"] = filtered
            self.critical_records.append(enriched_record)

    def save_csv(self, path: str):
        if not self.critical_records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
            'output_diff', 'critical_analysis'
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.critical_records:
                row = rec.copy()
                row['critical_analysis'] = json.dumps(row.get('critical_analysis', {}), ensure_ascii=False)
                writer.writerow(row)
            f.flush()

    def save_json(self, path: str):
        if not self.critical_records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.critical_records, f, indent=2, ensure_ascii=False)
            f.flush()
