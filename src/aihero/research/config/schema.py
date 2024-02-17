"""Schema for the configuration file, including project, dataset, model, and training settings."""
from typing import Optional, List, Enum
from pydantic import BaseModel, validator
from transformers import TrainingArguments
from peft import LoraConfig


class Project(BaseModel):
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


class Dataset(BaseModel):
    """Describes a dataset including its name, type, and associated task."""

    name: str
    type: Type
    task: Task


class BaseModelConfig(BaseModel):
    """Configuration for a base model including its name and type."""

    name: str
    type: Type


class Model(BaseModel):
    """Combines base and optional output model configurations."""

    base: BaseModelConfig
    output: Optional[BaseModelConfig] = None


class Trainer(BaseModel):
    """Configuration for model training, including packing and sequence length settings."""

    packing: bool
    max_seq_length: int


class Generator(BaseModel):
    """Configuration for generation tasks, similar to Trainer."""

    packing: bool
    max_seq_length: int


class Tokenizer(BaseModel):
    """Optional configuration for tokenizers, including additional tokens."""

    additional_tokens: Optional[List[str]] = None


class Freeze(BaseModel):
    """Miscellaneous optional configurations for training, like embedding freezing."""

    freeze_embed: Optional[bool] = None
    n_freeze: Optional[int] = None


class Training(BaseModel):
    """Comprehensive configuration for training, including task, dataset, model, and others."""

    task: Task
    dataset: Dataset
    model: Model
    trainer: Trainer
    sft: TrainingArguments
    peft: Optional[LoraConfig] = None
    quantized: Optional[bool] = None
    tokenizer: Optional[Tokenizer] = None
    freeze: Optional[Freeze] = None

    @validator("quantized")
    def check_quantized_with_peft(cls, v, values, **kwargs):
        """Ensure 'quantized' is True only if 'peft' is provided."""
        if v and not values.get("peft"):
            raise ValueError("'quantized' can be True only if 'peft' is provided.")
        return v


class BatchInference(BaseModel):
    """Configuration for batch inference, including task, dataset, and model details."""

    task: Task
    dataset: Dataset
    model: Model
    generator: Generator


class Config(BaseModel):
    """Root configuration combining project, training, and batch inference settings with exclusivity validation."""

    project: Project
    training: Optional[Training] = None
    batch_inference: Optional[BatchInference] = None

    @validator("training", "batch_inference", pre=True, each_item=True)
    def check_exclusivity(cls, v, values, **kwargs):
        """Ensure exclusivity between training and batch inference configurations."""
        if "training" in values and "batch_inference" in values:
            raise ValueError("Config can include either training or batch_inference, not both.")
        return v
