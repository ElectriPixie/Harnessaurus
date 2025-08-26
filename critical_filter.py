import csv
import json
import os
from typing import Optional, Dict, Any, List
from data_structures import Record, Output


class CriticalRecordFilter:
    """
    Filter Record objects to identify 'critical' plugin flags indicating suspicious or problematic outputs.
    Supports:
    - Boolean flags based on key suffixes (e.g. 'flagged', 'detected', etc.)
    - Numeric thresholds on keys containing substrings like 'score', 'count', etc.
    - Special handling for refusal_detector plugin (always included if non-empty)
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
        self.critical_records: List[Record] = []  # store Record objects directly

    def _is_critical_boolean(self, key: str, value: Any) -> bool:
        return isinstance(value, bool) and any(key.lower().endswith(suffix) for suffix in self.CRITICAL_BOOL_SUFFIXES) and value

    def _is_critical_numeric(self, plugin_name: str, key: str, value: Any) -> bool:
        if isinstance(value, (int, float)) and any(substr in key.lower() for substr in self.CRITICAL_NUMERIC_SUBSTRINGS):
            threshold = self.plugin_thresholds.get(plugin_name, self.DEFAULT_NUMERIC_THRESHOLD)
            return value >= threshold
        return False

    def _merge_output_analysis(self, outputs: List[Output]) -> Dict[str, Any]:
        """
        Merge analysis dictionaries from multiple Output objects into a single dict.
        """
        merged: Dict[str, Any] = {}
        for o in outputs:
            if o.analysis and isinstance(o.analysis, dict):
                for plugin_name, plugin_data in o.analysis.items():
                    if plugin_name in merged and isinstance(merged[plugin_name], dict) and isinstance(plugin_data, dict):
                        merged[plugin_name].update(plugin_data)
                    else:
                        merged[plugin_name] = plugin_data
        return merged

    def filter_record(self, record: Record) -> Dict[str, Dict[str, Any]]:
        """
        Returns only the suspicious parts of the record for each plugin.
        """
        filtered = {"analysis_clean": {}, "analysis_mutated": {}}

        merged_clean = self._merge_output_analysis(record.clean_outputs)
        merged_mutated = self._merge_output_analysis(record.mutated_outputs)

        for analysis_key, merged_analysis in zip(filtered.keys(), [merged_clean, merged_mutated]):
            filtered[analysis_key] = {}
            for plugin_name, plugin_data in merged_analysis.items():
                if isinstance(plugin_data, bool):
                    plugin_data = {"flagged": plugin_data}

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

    def is_critical(self, record: Record) -> bool:
        """
        True if record has any suspicious plugin outputs.
        """
        filtered = self.filter_record(record)
        return bool(filtered["analysis_clean"] or filtered["analysis_mutated"])

    def add_record(self, record: Record):
        """
        Store the Record object if it is critical.
        """
        if self.is_critical(record):
            self.critical_records.append(record)

    def save_csv(self, path: str):
        """
        Serialize critical Record objects to CSV.
        """
        if not self.critical_records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fieldnames = [
            'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
            'critical_analysis'
        ]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.critical_records:
                row = {
                    "original_prompt": rec.original_prompt,
                    "mutated_prompt": rec.mutated_prompt,
                    "clean_output": [o.raw_output for o in rec.clean_outputs],
                    "mutated_output": [o.raw_output for o in rec.mutated_outputs],
                    "critical_analysis": json.dumps(self.filter_record(rec), ensure_ascii=False)
                }
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def save_json(self, path: str):
        """
        Serialize critical Record objects to JSON.
        """
        if not self.critical_records:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        serialized = []
        for rec in self.critical_records:
            serialized.append({
                "original_prompt": rec.original_prompt,
                "mutated_prompt": rec.mutated_prompt,
                "clean_output": [o.raw_output for o in rec.clean_outputs],
                "mutated_output": [o.raw_output for o in rec.mutated_outputs],
                "critical_analysis": self.filter_record(rec)
            })
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
