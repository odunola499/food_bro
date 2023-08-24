import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)

import wandb
from getpass import getpass
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from huggingface_hub import login

huggingface_access_token = getpass("enter huggingface access token: ")
login(token=huggingface_access_token)
wandb_access_token = getpass("enter wandb access token: ")
wandb.login(key = wandb_access_token)

model_name = "meta-llama/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-odunola-foodie"
dataset_name = "odunola/foodiesdataset"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 32
per_device_eval_batch_size = 32
gradient_accumulation_steps = 2
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 25
max_seq_length = None
packing = False

#load dataset plus preprocessing
dataset = load_dataset(dataset_name)
dataset = dataset['train']
dataset = dataset.shuffle(seed = 42)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
test_dataset = dataset['test']

train_dataset = train_dataset.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['recipe'])]}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'text': [prompt + response for prompt, response in zip(examples['prompt'], examples['recipe'])]}, batched=True)


#loading model and lora configs

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type, #halfs the size of the model
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    #device_map=device_map
    #device_map =  {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 'cpu'}
    device_map = 'auto'
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

#training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=20  # Evaluate every 20 steps
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)
trainer.push_to_hub('Training complete!')

