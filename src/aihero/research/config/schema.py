"""Schema for the configuration file, including project, dataset, model, and training settings."""
import yaml
from typing import Optional, List
from pydantic import BaseModel, validator
from pydantic import ValidationError
from enum import Enum


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


class DatasetConfig(BaseModel, use_enum_values=True):
    """Describes a dataset including its name, type, and associated task."""

    name: str
    type: Type
    task: Task


class ModelConfig(BaseModel, use_enum_values=True):
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

    additional_tokens: List[str]


class FreezeExtras(BaseModel):
    """Miscellaneous optional configurations for training, like embedding freezing."""

    freeze_embed: Optional[bool] = None
    n_freeze: Optional[int] = None


class EvalExtras(BaseModel):
    """Extra configs and code for running tests and metrics."""

    size: Optional[int] = None
    randomize: Optional[bool] = None
    tests: Optional[str] = None
    metrics: Optional[str] = None


class SFT(BaseModel):
    """SFT Config."""

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    max_steps: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: Optional[dict] = None
    logging_strategy: str
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    bf16: Optional[bool] = None
    optim: Optional[str] = None
    max_grad_norm: Optional[float] = None


class PEFT(BaseModel):
    """PEFT Config."""

    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    task_type: str
    target_modules: List[str]
    quantized: Optional[bool] = None


class TrainingJob(BaseModel, use_enum_values=True):
    """Comprehensive configuration for training, including task, dataset, model, and others."""

    project: ProjectConfig
    task: Task
    dataset: DatasetConfig
    base: ModelConfig
    output: Optional[ModelConfig] = None
    trainer: TrainerExtras
    sft: SFT
    peft: Optional[PEFT] = None
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
            print(e.errors())
            raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            raise SystemExit(1)


class ServingService(BaseModel, use_enum_values=True):
    """Configuration for batch inference, including task, dataset, and model details."""

    project: ProjectConfig
    task: Task
    model: ModelConfig

    @staticmethod
    def load(file_path: str) -> "ServingService":
        """Load and validate a configuration file."""
        try:
            with open(file_path, "r") as file:
                data = yaml.safe_load(file)
                config = ServingService(**data)
                return config
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise SystemExit(1)
        except ValidationError as e:
            print(f"Validation error: {e}")
            print(e.errors())
            raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            raise SystemExit(1)


class BatchInferenceJob(BaseModel, use_enum_values=True):
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
            print(e.errors())
            raise SystemExit(1)
        except yaml.YAMLError as e:
            print(f"YAML parsing error: {e}")
            raise SystemExit(1)
