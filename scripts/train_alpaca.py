import wandb
"""
NOtes to self
1. remember to not use the fasdt version of AutoTOkenizer
"""
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset
from peft import PeftConfig, get_peft_model,LoraConfig
from torch import nn
dataset = load_dataset('tatsu-lab/alpaca', split = 'train').train_test_split(test_size =0.1)

train_dataset = dataset['train']
test_dataset = dataset['test']
model_id = 'openlm-research/open_llama_7b_v2'
wandb.login()
optim = 'adamw_torch'
model_max_length = 1000

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

peft_config = LoraConfig(
r = 16,
lora_alpha = 32,
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
bias = None,
task_type = "CAUSAL_LM"
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = 'auto'
)

for param in base_model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

base_model.gradient_checkpointing_enable()  # reduce number of stored activations
base_model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
base_model.lm_head = CastOutputToFloat(base_model.lm_head)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text']), batched = True)
tokenized_eval_dataset = test_dataset.map(lambda examples: tokenizer(examples['text']), batched = True)

trainer = Trainer(
  model = model,
  train_dataset = tokenized_train_dataset,
  eval_dataset = tokenized_eval_dataset,
  args =TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps = 100,
    max_steps = 1000,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 50,
    output_dir = 'outputs',
    optim = optim
  ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
)

trainer.train()
