import json
import os

class CriticalRecordFilter:
    def __init__(self):
        self.critical_records = []

    def is_critical(self, record: dict) -> bool:
        """
        Determines if a record is critical based on flags or suspicious indicators
        in either clean or mutated analysis.

        Returns True if any flagged or suspicious key is True.
        """
        if not record:
            return False

        suspicious_keys = [
            'flagged',
            'hidden_motivation_suspected',
            'evaluation_awareness_detected',
            'sandbagging_suspected',
            'reward_hacking_suspected',
            'sabotage_suspected'
        ]

        def check_analysis(analysis):
            if not isinstance(analysis, dict):
                return False
            for plugin_result in analysis.values():
                if not isinstance(plugin_result, dict):
                    continue
                for key in suspicious_keys:
                    if plugin_result.get(key):
                        return True
            return False

        clean_critical = check_analysis(record.get('analysis_clean'))
        mutated_critical = check_analysis(record.get('analysis_mutated'))

        return clean_critical or mutated_critical

    def add_record(self, record: dict):
        if self.is_critical(record):
            self.critical_records.append(record)

    def save_critical_csv(self, filepath: str):
        """
        Save critical records to CSV file with selected fields.
        """
        import csv

        if not self.critical_records:
            print("[CriticalRecordFilter] No critical records to save.")
            return

#        fieldnames = [
#            'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
#            'output_diff', 'critical_analysis', 'analysis_clean', 'analysis_mutated'
#        ]
        fieldnames = [
            'original_prompt', 'mutated_prompt', 'clean_output', 'mutated_output',
            'critical_analysis', 'analysis_clean', 'analysis_mutated'
        ]


        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for rec in self.critical_records:
                    critical_analysis = rec.get('critical_analysis', {}) if isinstance(rec, dict) else {}
                    analysis_clean = rec.get('analysis_clean', {}) if isinstance(rec, dict) else {}
                    analysis_mutated = rec.get('analysis_mutated', {}) if isinstance(rec, dict) else {}

                    row = rec.copy()

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

                    filtered_row = {k: row.get(k, '') for k in fieldnames}
                    writer.writerow(filtered_row)
            print(f"[CriticalRecordFilter] Successfully saved critical CSV: {filepath}")
        except Exception as e:
            print(f"[CriticalRecordFilter] Error saving critical CSV: {e}")

    def save_critical_json(self, filepath: str):
        """
        Save critical records to JSON file.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.critical_records, f, indent=2, ensure_ascii=False)
            print(f"[CriticalRecordFilter] Successfully saved critical JSON: {filepath}")
        except Exception as e:
            print(f"[CriticalRecordFilter] Error saving critical JSON: {e}")