class CriticalRecordFilter:
    # Default numeric threshold for flagging scores
    DEFAULT_NUMERIC_THRESHOLD = 1

    def __init__(self, plugin_thresholds=None):
        # Per-plugin thresholds can be customized here
        self.plugin_thresholds = plugin_thresholds or {}

    def is_value_critical(self, key, value, plugin_name):
        # Flag if boolean True
        if isinstance(value, bool) and value:
            return True
        # Flag if numeric and above threshold (plugin-specific or default)
        if isinstance(value, (int, float)):
            threshold = self.plugin_thresholds.get(plugin_name, self.DEFAULT_NUMERIC_THRESHOLD)
            if value >= threshold:
                return True
        return False

    def filter_plugin_result(self, plugin_name, plugin_data):
        # Keep only fields that are flagged/critical
        critical_data = {}
        for k, v in plugin_data.items():
            if self.is_value_critical(k, v, plugin_name):
                critical_data[k] = v
        return critical_data

    def filter_section(self, analysis_section):
        # Filter all plugins in one section (clean or mutated)
        filtered = {}
        for plugin_name, plugin_data in analysis_section.items():
            filtered_data = self.filter_plugin_result(plugin_name, plugin_data)
            if filtered_data:
                filtered[plugin_name] = filtered_data
        return filtered

    def filter_record(self, record):
        # Apply filtering for both clean and mutated analysis
        filtered_clean = self.filter_section(record.get("analysis_clean", {}))
        filtered_mutated = self.filter_section(record.get("analysis_mutated", {}))
        return {
            "analysis_clean": filtered_clean,
            "analysis_mutated": filtered_mutated,
        }