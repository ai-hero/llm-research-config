# llm-research-config
Repo for Pydantic Data Models for the configs driving the research.

These classes will be used for validating yaml files that run fine tuning jobs.

For example:
```yaml
project:
  name: "mmlu"

task: "completion"

dataset:
  name: "fine-tuning-research/mmlu"
  type: "s3"
  task: "completion"

base:
  name: "meta-llama/Llama-2-7b-hf"
  type: "huggingface"

output:
  name: "rparundekar/llama2-7b-mmlu"
  type: "huggingface"

trainer:
  packing: false
  max_seq_length: 512

sft:
  per_device_train_batch_size: 24
  per_device_eval_batch_size: 24
  bf16: true
  learning_rate: 0.0002
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  max_steps: 500
  optim: "paged_adamw_32bit"
  max_grad_norm: 0.3
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  logging_strategy: "steps"
  logging_steps: 5
  evaluation_strategy: "steps"
  eval_steps: 100

freeze:
  freeze_embed: true
  n_freeze: 24
```
