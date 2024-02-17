"""Schema for the configuration file, including project, dataset, model, and training settings."""
import yaml
from typing import Optional, List, Enum
from pydantic import BaseModel, validator
from transformers import TrainingArguments
from pydantic import ValidationError
from peft import LoraConfig


class ProjectConfig(BaseModel):
    """Represents a project with a name."""

    name: str


class Task(str, Enum):
    """Defines the types of tasks for datasets and models."""

    TEXT = "text"
    COMPLETION = "completion"
    RAG = "rag"


class Type(str, Enum):
    """Enumerates the storage types for datasets and models."""

    S3 = "s3"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class DatasetConfig(BaseModel):
    """Describes a dataset including its name, type, and associated task."""

    name: str
    type: Type
    task: Task


class ModelConfig(BaseModel):
    """Configuration for a base model including its name and type."""

    name: str
    type: Type


class TrainerExtras(BaseModel):
    """Configuration for model training, including packing and sequence length settings."""

    packing: bool
    max_seq_length: int


class GeneratorExtras(BaseModel):
    """Configuration for generation tasks, similar to Trainer."""

    packing: bool
    max_seq_length: int


class TokenizerExtras(BaseModel):
    """Optional configuration for tokenizers, including additional tokens."""

    additional_tokens: Optional[List[str]] = None


class FreezeExtras(BaseModel):
    """Miscellaneous optional configurations for training, like embedding freezing."""

    freeze_embed: Optional[bool] = None
    n_freeze: Optional[int] = None


class EvalExtras(BaseModel):
    size: int = 100
    tests: Optional[str] = None
    metrics: Optional[str] = None


class TrainingJob(BaseModel):
    """Comprehensive configuration for training, including task, dataset, model, and others."""

    project: ProjectConfig
    task: Task
    dataset: DatasetConfig
    base: ModelConfig
    output: Optional[ModelConfig] = None
    trainer: TrainerExtras
    sft: TrainingArguments
    peft: Optional[LoraConfig] = None
    quantized: Optional[bool] = None
    tokenizer: Optional[TokenizerExtras] = None
    freeze: Optional[FreezeExtras] = None
    eval: Optional[EvalExtras] = None

    @validator("quantized")
    def check_quantized_with_peft(cls, v, values, **kwargs):
        """Ensure 'quantized' is True only if 'peft' is provided."""
        if v and not values.get("peft"):
            raise ValueError("'quantized' can be True only if 'peft' is provided.")
        return v

    @staticmethod
    def load(file_path: str) -> "TrainingJob":
        """Load and validate a configuration file."""
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                config = TrainingJob(**data)
                return config
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise SystemExit(1)
        except ValidationError as e:
            print(f"Validation error: {e}")
            raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            raise SystemExit(1)


class BatchInferenceJob(BaseModel):
    """Configuration for batch inference, including task, dataset, and model details."""

    project: ProjectConfig
    task: Task
    dataset: DatasetConfig
    model: ModelConfig
    generator: GeneratorExtras
    eval: Optional[EvalExtras] = None

    @staticmethod
    def load(file_path: str) -> "BatchInferenceJob":
        """Load and validate a configuration file."""
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                config = BatchInferenceJob(**data)
                return config
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise SystemExit(1)
        except ValidationError as e:
            print(f"Validation error: {e}")
            raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            raise SystemExit(1)
