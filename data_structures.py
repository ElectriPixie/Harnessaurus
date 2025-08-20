# data_structures.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from typing import Literal

OutputType = Literal["single", "multi"]

@dataclass
class PromptSet:
    """Wrapper for a collection of Prompt objects (variable-length 'list')."""
    prompts: List[Prompt] = field(default_factory=list)
    output_type: OutputType = "multi"  # defaults to multi since it's a set
    tags: dict = field(default_factory=dict)  # optional structured metadata for the set

    def add_prompt(self, prompt: Prompt) -> None:
        """Append a single Prompt to the set."""
        self.prompts.append(prompt)

    def extend_prompts(self, prompt_list: List[Prompt]) -> None:
        """Add multiple Prompts to the set."""
        self.prompts.extend(prompt_list)

    def __len__(self):
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index]

@dataclass
class Prompt:
    """Minimal prompt representation."""
    prompt_list: List[str]                  # e.g., ["original", "probe"]
    has_context: bool = False               # True if multiple context prompts
    output_type: OutputType = "single"  # restrict to "single" or "multi"
    tags: Dict[str, any] = field(default_factory=dict)  # structured metadata
    plugin_meta: dict = field(default_factory=dict)  # metadata per plugin


@dataclass
class RunPrompt:
    """Wrapper for prompt + execution settings."""
    prompt_obj: Prompt
    iterator: int = 1
    flip_negate: bool = False
    max_tokens_per_chunk: int = 256
    max_iterations: int = 10
    loop: bool = True
    max_mutations: int = 1
    mutators: Optional[List[str]] = None
    include_mutated_output: bool = True
    rerun_clean_prompt: bool = False
    run_dir: Optional[str] = None
    last_mutator_name: Optional[str] = None
    user_generator: Optional[str] = None


@dataclass
class Output:
    """Represents a single model output with optional analysis."""
    prompt: Prompt                           # Original Prompt
    raw_output: str                          # Model output text
    analysis: Dict[str, Any] = field(default_factory=dict)  # Plugin analysis
    mutation_iteration: int = 0
    run_dir: Optional[str] = None


@dataclass
class Record:
    """Full record of a prompt run, including mutated outputs."""
    original_prompt: str
    mutated_prompt: str = ""
    clean_output: Optional[Output] = None
    mutated_output: Optional[Output] = None
    mutation_iteration: int = 0
    run_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flatten Record for JSON/CSV export."""
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