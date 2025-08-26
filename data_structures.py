from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from typing import Literal

# Type for output
OutputType = Literal["single", "multi"]

# ----------------------------
# Minimal Prompt representation
# ----------------------------
@dataclass
class Prompt:
    prompt_list: List[Dict[str, Any]] = field(default_factory=list)
    has_context: bool = False
    output_type: OutputType = "single"
    tags: Dict[str, Any] = field(default_factory=dict)
    plugin_meta: Dict[str, Any] = field(default_factory=dict)

    def _get_item_text(self, item) -> str:
        if isinstance(item, dict):
            # Ensure we always return a string, even if 'text' is None
            return str(item.get("text", "") or "")
        elif isinstance(item, str):
            return item
        return str(item)  # fallback for other types

    @property
    def output_text(self) -> str:
        texts = []
        for item in self.prompt_list:
            text = self._get_item_text(item)
            if text.strip():
                texts.append(text)
        return "\n".join(texts)
# ----------------------------
# Collection of Prompts
# ----------------------------
@dataclass
class PromptSet:
    """Wrapper for a collection of Prompt objects."""
    prompts: List["Prompt"] = field(default_factory=list)  # Forward reference
    output_type: OutputType = "multi"
    tags: Dict[str, Any] = field(default_factory=dict)

    def add_prompt(self, prompt: "Prompt") -> None:
        self.prompts.append(prompt)

    def extend_prompts(self, prompt_list: List["Prompt"]) -> None:
        self.prompts.extend(prompt_list)

    def __len__(self):
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

    def all_output_texts(self) -> List[str]:
        return [p.output_text for p in self.prompts]

# ----------------------------
# Output from model
# ----------------------------
@dataclass
class Output:
    prompt: "Prompt"
    raw_output: str  # always contains the full legacy string
    final: Optional[str] = None  # explicitly store the "final" content
    analysis: Dict[str, Any] = field(default_factory=dict)
    mutation_iteration: int = 0
    run_dir: Optional[str] = None
    channels: Dict[str, str] = field(default_factory=dict)

    @property
    def output_text(self) -> str:
        """
        For legacy detectors: return the raw string.
        """
        return self.raw_output

    def set_channels(self, channels_dict: dict):
        """
        Populate channels dictionary safely. Also optionally extracts 'final'.
        """
        if not isinstance(channels_dict, dict):
            channels_dict = {}
        
        # Ensure all expected channels exist
        safe_channels = {
            "final": "",
            "analysis": "",
            "commentary": ""
        }
        for key in safe_channels:
            try:
                if key in channels_dict and channels_dict[key] is not None:
                    safe_channels[key] = str(channels_dict[key])
            except Exception:
                # ignore any malformed entry
                continue

        self.channels = safe_channels

        # Auto-fill final if present and not set
        if not hasattr(self, "final") or not self.final:
            self.final = safe_channels.get("final", "")

    def get_channel(self, name: str) -> str:
        """
        Safely retrieve a channel's text.
        """
        return self.channels.get(name, "")


# ----------------------------
# Record of prompt execution
# ----------------------------
@dataclass
class Record:
    original_prompt: str
    mutated_prompt: str = ""
    clean_outputs: List[Output] = field(default_factory=list)
    mutated_outputs: List[Output] = field(default_factory=list)
    mutation_iteration: int = 0
    run_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt or "",
            "mutated_prompt": self.mutated_prompt or "",
            "clean_output": [o.raw_output for o in self.clean_outputs],
            "mutated_output": [o.raw_output for o in self.mutated_outputs],
            "clean_channels": [o.channels for o in self.clean_outputs],
            "mutated_channels": [o.channels for o in self.mutated_outputs],
            "analysis_clean": [o.analysis for o in self.clean_outputs],
            "analysis_mutated": [o.analysis for o in self.mutated_outputs],
            "mutation_iteration": self.mutation_iteration,
            "run_dir": self.run_dir,
        }

# ----------------------------
# RunPrompt wrapper (forward refs for plugins)
# ----------------------------
@dataclass
class RunPrompt:
    prompt_obj: "Prompt"
    iterator: int = 1
    flip_negate: bool = False
    max_tokens_per_chunk: int = 256
    max_iterations: int = 10
    loop: bool = True
    use_mutators: Optional[List[str]] = None
    max_mutations: int = 1
    mutators: Optional[List[str]] = None
    detector_plugins: Optional[List["DetectorPlugin"]] = None
    mutator_plugins: Optional[List["MutatorPlugin"]] = None
    logger_plugins: Optional[List["BasePlugin"]] = None
    generator_plugins: Optional[List["GeneratorBase"]] = None
    use_mutated: bool = True
    rerun_clean_prompt: bool = False
    run_dir: Optional[str] = None
    use_generator: Optional[str] = None
    legacy_mode: bool = False  # <-- new flag

