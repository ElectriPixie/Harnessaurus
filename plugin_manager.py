# plugin_manager.py
from typing import List, Dict, Optional
from plugin_base import PluginBase, MutatorPlugin, DetectorPlugin
from data_structures import Prompt, Output, Record
from utils import split_into_channels, debug_print

DEBUG=True

class PluginManager:
    def __init__(
        self,
        mutators: Optional[List[MutatorPlugin]] = None,
        detectors: Optional[List[DetectorPlugin]] = None,
        loggers: Optional[List[PluginBase]] = None,
        channel_map: Optional[Dict[str, List[str]]] = None
    ):
        self.mutators = mutators or []
        self.detectors = detectors or []
        self.loggers = loggers or []
        self.channel_map = channel_map or {}

    # ---------------- Prompt Processing ----------------
    def process_prompt(
        self,
        prompt_obj: Prompt,
        plugins_to_apply: Optional[List[str]] = None,
    ) -> Prompt:
        """Apply mutators to the prompt."""
        debug_print(DEBUG, f"[DEBUG] Original prompt: {prompt_obj.prompt_list}")
        debug_print(DEBUG, f"[DEBUG] plugins_to_apply: {plugins_to_apply}")

        # Resolve plugins to apply
        if plugins_to_apply is None:
            active_plugins = []
        elif plugins_to_apply == []:
            active_plugins = self.mutators
        else:
            name_to_plugin = {p.__class__.__name__: p for p in self.mutators}
            active_plugins = [name_to_plugin[n] for n in plugins_to_apply if n in name_to_plugin]
            missing = [n for n in plugins_to_apply if n not in name_to_plugin]
            if missing:
                print(f"[WARNING] Mutator plugins not found: {missing}")

        # Apply each mutator in order
        for plugin in active_plugins:
            debug_print(DEBUG, f"[DEBUG] Applying mutator: {plugin.__class__.__name__}")
            prompt_obj = plugin.process_prompt(prompt_obj=prompt_obj)

            # Normalize prompt_list items
            for i, item in enumerate(prompt_obj.prompt_list):
                if isinstance(item, str):
                    prompt_obj.prompt_list[i] = {"text": item}
                elif isinstance(item, dict) and "text" not in item:
                    prompt_obj.prompt_list[i]["text"] = str(item)

            debug_print(DEBUG, f"[DEBUG] Prompt after {plugin.__class__.__name__}: {prompt_obj.prompt_list}")

        return prompt_obj

    # ---------------- Output Processing ----------------
    def process_output(
        self,
        prompt_obj: Prompt,
        output_obj: Output,
        plugins_to_apply: Optional[List[str]] = None
    ) -> Output:
        """Run detectors on the output object."""
        if not output_obj.channels:
            output_obj.channels = split_into_channels(output_obj)

        # Resolve detectors
        if plugins_to_apply is None:
            active_detectors = []
        elif plugins_to_apply == []:
            active_detectors = self.detectors
        else:
            name_to_plugin = {d.__class__.__name__: d for d in self.detectors}
            active_detectors = [name_to_plugin[n] for n in plugins_to_apply if n in name_to_plugin]
            missing = [n for n in plugins_to_apply if n not in name_to_plugin]
            if missing:
                print(f"[WARNING] Detector plugins not found: {missing}")

        if output_obj.analysis is None:
            output_obj.analysis = {}

        for detector in active_detectors:
            detector_name = detector.__class__.__name__
            channels_for_detector = self.channel_map.get(detector_name)

            if channels_for_detector:
                detector_input_text = "\n".join(
                    output_obj.channels.get(ch, "") for ch in channels_for_detector
                )
            else:
                detector_input_text = output_obj.raw_output

            detector_input_obj = Output(prompt=prompt_obj, raw_output=detector_input_text)
            analysis_result_obj = detector.process_output(prompt_obj=prompt_obj, output_obj=detector_input_obj)
            output_obj.analysis[detector_name] = analysis_result_obj.analysis.get(detector_name)

        return output_obj

    # ---------------- Logging ----------------
    def process_log(self, record: Record):
        for logger in self.loggers:
            if hasattr(logger, "on_log"):
                logger.on_log(record)

    # ---------------- Utility ----------------
    def detectors_by_name(self) -> Dict[str, DetectorPlugin]:
        return {d.__class__.__name__: d for d in self.detectors}
