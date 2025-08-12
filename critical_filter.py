import csv
import json
import os
from typing import Optional, Dict, Any


class CriticalRecordFilter:
    """
    Filter records to identify 'critical' plugin flags indicating suspicious or problematic outputs.
    Supports:
    - Boolean flags based on key suffixes (e.g. 'flagged', 'detected', etc.)
    - Numeric thresholds on keys containing substrings like 'score', 'count', etc.
    - Special handling for refusal_detector plugin (always included if non-empty)
    
    Keeps a list of critical records with enriched critical_analysis data.
    Provides saving of critical records to CSV and JSON files.
    """

    CRITICAL_BOOL_SUFFIXES = [
        "flagged",
        "suspected",
        "detected",
        "error",
        "alert",
        "warning",
        "shift",      # e.g. possible_evasive_shift
        "patterns",   # e.g. suspicious_patterns
    ]

    CRITICAL_NUMERIC_SUBSTRINGS = [
        "score",
        "count",
        "level",
        "severity",
    ]

    DEFAULT_NUMERIC_THRESHOLD = 5

    REFUSAL_DETECTOR_PLUGIN = "refusal_detector.RefusalDetector"

    def __init__(self, plugin_thresholds: Optional[Dict[str, float]] = None, debug: bool = False):
        """
        Args:
            plugin_thresholds: Optional dict mapping plugin names to numeric thresholds.
            debug: Enable debug printing.
        """
        self.plugin_thresholds = plugin_thresholds or {}
        self.critical_records = []
        self.debug = False

    def _debug_print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def _is_critical_boolean(self, key: str, value: Any) -> bool:
        """Check if boolean plugin flag is critical."""
        if isinstance(value, bool):
            match = any(key.lower().endswith(suffix) for suffix in self.CRITICAL_BOOL_SUFFIXES)
            self._debug_print(f"[is_critical_boolean] key={key}, value={value}, match={match}")
            return match and value
        return False

    def _is_critical_numeric(self, plugin_name: str, key: str, value: Any) -> bool:
        """Check if numeric plugin flag is critical based on thresholds."""
        if isinstance(value, (int, float)):
            if any(substr in key.lower() for substr in self.CRITICAL_NUMERIC_SUBSTRINGS):
                threshold = self.plugin_thresholds.get(plugin_name, self.DEFAULT_NUMERIC_THRESHOLD)
                triggered = value >= threshold
                self._debug_print(
                    f"[is_critical_numeric] plugin={plugin_name}, key={key}, value={value}, "
                    f"threshold={threshold}, triggered={triggered}"
                )
                return triggered
        return False

    def filter_record(self, record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract critical plugin flags from a record's analysis_clean and analysis_mutated fields.
        Special case for refusal_detector plugin: always include if any truthy value.

        Returns filtered dict with only critical findings.
        """
        filtered = {"analysis_clean": {}, "analysis_mutated": {}}

        for analysis_key in filtered.keys():
            analysis = record.get(analysis_key, {})
            filtered[analysis_key] = {}

            for plugin_name, plugin_data in analysis.items():
                # Defensive: if plugin_data is boolean, wrap it for uniformity
                if isinstance(plugin_data, bool):
                    plugin_data = {"flagged": plugin_data}
                    self._debug_print(f"[filter_record] Wrapped boolean plugin_data for {plugin_name}: {plugin_data}")

                # Special case refusal_detector: include if any truthy value
                if plugin_name == self.REFUSAL_DETECTOR_PLUGIN:
                    if isinstance(plugin_data, dict) and any(plugin_data.values()):
                        filtered[analysis_key][plugin_name] = plugin_data
                        self._debug_print(f"[filter_record] Included refusal_detector in {analysis_key}")
                    continue

                # Skip malformed plugin data
                if not isinstance(plugin_data, dict):
                    self._debug_print(f"[filter_record] Skipping plugin {plugin_name} in {analysis_key} due to bad data: {plugin_data}")
                    continue

                suspicious = {}
                for key, value in plugin_data.items():
                    if self._is_critical_boolean(key, value) or self._is_critical_numeric(plugin_name, key, value):
                        suspicious[key] = value

                if suspicious:
                    filtered[analysis_key][plugin_name] = suspicious
                    self._debug_print(f"[filter_record] Plugin {plugin_name} suspicious keys in {analysis_key}: {list(suspicious.keys())}")

            self._debug_print(f"[filter_record] Filtered plugins in {analysis_key}: {list(filtered[analysis_key].keys())}")

        return filtered

    def is_critical(self, record: Dict[str, Any]) -> bool:
        """
        Return True if record has any critical flags detected in clean or mutated analysis.
        """
        filtered = self.filter_record(record)
        critical = bool(filtered["analysis_clean"]) or bool(filtered["analysis_mutated"])
        self._debug_print(f"[is_critical] critical={critical}")
        return critical

    def add_record(self, record: Dict[str, Any]):
        """
        Add record to internal list if critical flags found.
        Enriches record with 'critical_analysis' field.
        """
        filtered = self.filter_record(record)
        if filtered["analysis_clean"] or filtered["analysis_mutated"]:
            enriched_record = dict(record)
            enriched_record["critical_analysis"] = filtered
            self.critical_records.append(enriched_record)
            self._debug_print(f"[add_record] Added critical record. Total now: {len(self.critical_records)}")
        else:
            self._debug_print("[add_record] Record not critical, skipping.")

    def save_csv(self, path: str):
        """
        Save all critical records to CSV.
        """
        if not self.critical_records:
            self._debug_print("[save_csv] No critical records to save.")
            return

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
                    row['critical_analysis'] = json.dumps(row.get('critical_analysis', {}), ensure_ascii=False, indent=2)
                    writer.writerow(row)
            self._debug_print(f"[save_csv] Saved {len(self.critical_records)} records to CSV at '{path}'")
        except Exception as e:
            self._debug_print(f"[save_csv] Exception while saving CSV: {e}")

    def save_json(self, path: str):
        """
        Save all critical records to JSON.
        """
        if not self.critical_records:
            self._debug_print("[save_json] No critical records to save.")
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.critical_records, f, indent=2, ensure_ascii=False)
            self._debug_print(f"[save_json] Saved {len(self.critical_records)} records to JSON at '{path}'")
        except Exception as e:
            self._debug_print(f"[save_json] Exception while saving JSON: {e}")
