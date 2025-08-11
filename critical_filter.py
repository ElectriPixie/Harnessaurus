import csv
import json
import os
from typing import Optional, Dict, Any

class CriticalRecordFilter:
    """
    Filter records to find 'critical' plugin flags indicating suspicious or problematic outputs.
    Special handling for refusal_detector plugin to always include if flagged/non-empty.

    Supports boolean flags and numeric thresholds.
    """

    # Suffixes for keys to consider boolean flags critical if True
    CRITICAL_BOOL_SUFFIXES = [
        "flagged",
        "suspected",
        "detected",
        "error",
        "alert",
        "warning",
        "shift",      # catch possible_evasive_shift
        "patterns",   # catch suspicious_patterns
    ]

    # Numeric key substrings to consider and default threshold to mark critical
    CRITICAL_NUMERIC_SUBSTRINGS = [
        "score",
        "count",
        "level",
        "severity",
    ]
    DEFAULT_NUMERIC_THRESHOLD = 5

    def __init__(self, plugin_thresholds: Optional[Dict[str, float]] = None, debug: bool = False):
        self.plugin_thresholds = plugin_thresholds or {}
        self.critical_records = []
        self.debug = True 
        self.refusal_detector_plugin = "RefusalDetector"

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def is_critical_boolean(self, key: str, value: Any) -> bool:
        """
        Check if a boolean plugin result key-value pair is critical.
        Only if value is True and key ends with any critical suffix.
        """
        if isinstance(value, bool):
            match = any(key.lower().endswith(suffix) for suffix in self.CRITICAL_BOOL_SUFFIXES)
            self._debug_print(f"[is_critical_boolean] key='{key}', value={value}, match={match}")
            return match and value
        return False

    def is_critical_numeric(self, plugin_name: str, key: str, value: Any) -> bool:
        """
        Check if a numeric plugin result key-value pair is critical based on thresholds.
        """
        if isinstance(value, (int, float)):
            substr_match = any(substr in key.lower() for substr in self.CRITICAL_NUMERIC_SUBSTRINGS)
            if substr_match:
                threshold = self.plugin_thresholds.get(plugin_name, self.DEFAULT_NUMERIC_THRESHOLD)
                triggered = value >= threshold
                self._debug_print(
                    f"[is_critical_numeric] plugin='{plugin_name}', key='{key}', value={value}, "
                    f"threshold={threshold}, triggered={triggered}"
                )
                return triggered
        return False

    def filter_record(self, record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Filter the record's analysis_clean and analysis_mutated fields to only include
        critical plugin flags or numeric values.

        The refusal_detector plugin is always included if non-empty or flagged.

        Args:
            record: Single test record dictionary containing plugin analyses.

        Returns:
            Filtered dict with same keys ('analysis_clean', 'analysis_mutated') mapping
            to dicts of plugin names to critical findings.
        """
        filtered = {"analysis_clean": {}, "analysis_mutated": {}}
        refused = {"analysis_clean": {}, "analysis_mutated": {}}

        for analysis_key in ["analysis_clean", "analysis_mutated"]:
            analysis = record.get(analysis_key, {})
            filtered[analysis_key] = {}

            for plugin_name, plugin_data in analysis.items():
                self._debug_print(f"[filter_record] Checking plugin: {plugin_name} with data: {plugin_data}")

                if plugin_name == self.refusal_detector_plugin:
                    self._debug_print(f"[filter_record] Found refusal detector plugin in {analysis_key}")
                    if isinstance(plugin_data, dict) and any(plugin_data.values()):
                        refused[analysis_key][plugin_name] = plugin_data
                        self._debug_print(
                            f"[filter_record] Included refusal_detector plugin in {analysis_key} with data: {plugin_data}"
                        )
                    else:
                        self._debug_print(
                            f"[filter_record] refusal_detector plugin in {analysis_key} has no truthy values; skipping"
                        )
                    continue

                # Skip if plugin_data is not dict (could be malformed)
                if not isinstance(plugin_data, dict):
                    self._debug_print(
                        f"[filter_record] Skipping plugin '{plugin_name}' in {analysis_key} because plugin_data is not dict: {plugin_data}"
                    )
                    continue

                suspicious_items = {}
                for key, value in plugin_data.items():
                    if self.is_critical_boolean(key, value) or self.is_critical_numeric(plugin_name, key, value):
                        suspicious_items[key] = value

                if suspicious_items:
                    filtered[analysis_key][plugin_name] = suspicious_items

                    self._debug_print(
                        f"[filter_record] Plugin '{plugin_name}' suspicious items in {analysis_key}: {suspicious_items}"
                    )

            self._debug_print(f"[filter_record] Filtered plugins in {analysis_key}: {list(filtered[analysis_key].keys())}")

        return filtered, refused

    def is_critical(self, record: Dict[str, Any]) -> bool:
        """
        Return True if the record has any critical plugin flags detected.
        """
        filtered, refused = self.filter_record(record)
        critical = bool(filtered["analysis_clean"]) or bool(filtered["analysis_mutated"])
        self._debug_print(f"[is_critical] Record critical={critical}")
        return critical

    def add_record(self, record: Dict[str, Any]):
        """
        Add record to critical_records if it has critical findings.
        """
        filtered, refused = self.filter_record(record)
        filtered["analysis_clean"]["RefusalDetector"] = refused["analysis_clean"]["RefusalDetector"]
        filtered["analysis_mutated"]["RefusalDetector"] = refused["analysis_mutated"]["RefusalDetector"]

        if filtered["analysis_clean"] or filtered["analysis_mutated"]:
            self._debug_print(
                f"[add_record] Adding critical record with plugins: "
                f"clean={list(filtered['analysis_clean'].keys())}, "
                f"mutated={list(filtered['analysis_mutated'].keys())}"
            )
            self.critical_records.append({
                "original_prompt": record.get("original_prompt"),
                "mutated_prompt": record.get("mutated_prompt"),
                "clean_output": record.get("clean_output"),
                "mutated_output": record.get("mutated_output"),
                "output_diff": record.get("output_diff"),
                "critical_analysis": filtered,
            })
            self._debug_print(f"[add_record] Total stored critical records: {len(self.critical_records)}")
        else:
            self._debug_print("[add_record] Not adding record; no critical plugins detected.")

    def save_csv(self, path: str):
        """
        Save all critical records to CSV file.
        """
        if not self.critical_records:
            self._debug_print("[save_csv] No critical records to save.")
            return

        self._debug_print(f"[save_csv] Saving {len(self.critical_records)} records to CSV at '{path}'")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'original_prompt',
                    'mutated_prompt',
                    'clean_output',
                    'mutated_output',
                    'output_diff',
                    'critical_analysis',
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in self.critical_records:
                    row = rec.copy()
                    row['critical_analysis'] = json.dumps(row['critical_analysis'], ensure_ascii=False, indent=2)
                    writer.writerow(row)
        except Exception as e:
            self._debug_print(f"[save_csv] Exception while saving CSV: {e}")

    def save_json(self, path: str):
        """
        Save all critical records to JSON file.
        """
        if not self.critical_records:
            self._debug_print("[save_json] No critical records to save.")
            return

        self._debug_print(f"[save_json] Saving {len(self.critical_records)} records to JSON at '{path}'")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.critical_records, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._debug_print(f"[save_json] Exception while saving JSON: {e}")
