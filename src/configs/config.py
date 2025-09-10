from enum import Enum
from pydantic import BaseModel as PBM
from pydantic import Field
from typing import Any, Dict, List, Optional

class Experiment(Enum):
    
    EXPERIMENT_INFERENCE = "EXPERIMENT_INFERENCE"
    EXPERIMENT_SANITIZATION = "EXPERIMENT_SANITIZATION"
    EXPERIMENT_EXPLICIT_SANITIZATION = "EXPERIMENT_EXPLICIT_SANITIZATION"
    EXPERIMENT_SANITIZATION_DECEPTION = "EXPERIMENT_SANITIZATION_DECEPTION"
    EXPERIMENT_POST_INFERENCE = "EXPERIMENT_POST_INFERENCE"
    EXPERIMENT_POST_INFERENCE_TRUTH_CONFIDENCE = "EXPERIMENT_POST_INFERENCE_TRUTH_CONFIDENCE"
    EXPERIMENT_LLM_UTILITY = "EXPERIMENT_LLM_UTILITY"
    EXPERIMENT_TOPIC = "EXPERIMENT_TOPIC"
    EXPERIMENT_TOPIC_PRIOR = "EXPERIMENT_TOPIC_PRIOR"
    EXPERIMENT_DECEPTION_POST_SANITIZATION_CONFIDENCE = "EXPERIMENT_DECEPTION_POST_SANITIZATION_CONFIDENCE"
    

class Task(Enum):
    REDDIT = "REDDIT"  # Reddit
    # Synthetic options
    SYNTHETIC = "SYNTHETIC"  # Synthetic Reddit comments generation

class ModelConfig(PBM):
    name: str = Field(description="Name of the model")
    tokenizer_name: Optional[str] = Field(
        None, description="Name of the tokenizer to use"
    )
    provider: str = Field(description="Provider of the model")
    dtype: str = Field(
        "float16", description="Data type of the model (only used for local models)"
    )
    device: str = Field(
        "auto", description="Device to use for the model (only used for local models)"
    )
    max_workers: int = Field(
        1, description="Number of workers (Batch-size) to use for parallel generation"
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the model upon generation",
    )
    model_template: str = Field(
        default="{prompt}",
        description="Template to use for the model (only used for local models)",
    )
    prompt_template: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the prompt"
    )
    submodels: List["ModelConfig"] = Field(
        default_factory=list, description="Submodels to use"
    )
    multi_selector: str = Field(
        default="majority", description="How to select the final answer"
    )

    def get_name(self) -> str:
        if self.name == "multi":
            return "multi" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.name == "chain":
            return "chain_" + "_".join(
                [submodel.get_name() for submodel in self.submodels]
            )
        if self.provider == "hf":
            return self.name.split("/")[-1]
        else:
            return self.name
        
        
class SYNTHETICConfig(PBM):
    path: str = Field(default=None, description='location of synthetic dataset')
    
    individual_prompts: bool = Field(
        default=False,
        description="Whether we want one prompt per attribute inferred or one for all.",
    )
    
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt to use"
    )
    
    num_of_guesses: int = Field(
        default=1,
        description='Number of guesses by the model'
    )
    
    reasoning: str = Field(
        default=False,
        description='Reasoning in the output of the model'
    )

class Config(PBM):
    task_config: SYNTHETICConfig = Field(default=None, description="Config for the task")
    
    gen_model: ModelConfig = Field(
        default=None, description="Model to use for generation, ignored for CHAT task"
    )
    task: Task = Field(
        default=None, description="Task to run", choices=list(Task.__members__.values())
    )
    
    