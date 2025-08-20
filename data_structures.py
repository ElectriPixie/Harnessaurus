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
    prompt_list: List[str]                     # e.g., ["original", "probe"]
    has_context: bool = False                  # True if multiple context prompts
    output_type: OutputType = "single"        # "single" or "multi"
    tags: Dict[str, Any] = field(default_factory=dict)
    plugin_meta: Dict[str, Any] = field(default_factory=dict)
    @property
    def output_text(self) -> str:
        return self.raw_output

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

# ----------------------------
# Output from model
# ----------------------------
@dataclass
class Output:
    prompt: "Prompt"
    raw_output: str
    analysis: Dict[str, Any] = field(default_factory=dict)
    mutation_iteration: int = 0
    run_dir: Optional[str] = None
    channels: Dict[str, str] = field(default_factory=dict)
    @property
    def output_text(self) -> str:
        return self.raw_output


# ----------------------------
# Record of prompt execution
# ----------------------------
@dataclass
class Record:
    original_prompt: str
    mutated_prompt: str = ""
    clean_output: Optional[Output] = None
    mutated_output: Optional[Output] = None
    mutation_iteration: int = 0
    run_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "mutated_prompt": self.mutated_prompt,
            "clean_output": self.clean_output.raw_output if self.clean_output else "",
            "mutated_output": self.mutated_output.raw_output if self.mutated_output else "",
            "mutation_iteration": self.mutation_iteration,
            "run_dir": self.run_dir,
            "analysis_clean": self.clean_output.analysis if self.clean_output else {},
            "analysis_mutated": self.mutated_output.analysis if self.mutated_output else {},
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
