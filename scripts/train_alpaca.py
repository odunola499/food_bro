import wandb
"""
NOtes to self
1. remember to not use the fasdt version of AutoTOkenizer
"""
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig
import wandb
from huggingface_hub import login
from trl import SFTTrainer

login(token = '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')

dataset = load_dataset('tatsu-lab/alpaca', split = 'train').train_test_split(test_size =0.2)

train_dataset = dataset['train']
test_dataset = dataset['test']
model_id = 'openlm-research/open_llama_3b_v2' #for testing. final model to be trained is a 7b parameter base model
wandb.login(key = 'bceff3fe9a5725b89b5986c88c7d6ba6d8d304a0')
optim = "paged_adamw_32bit"
model_max_length = 1000

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

peft_config = LoraConfig(
r = 16,
lora_alpha = 32,
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
bias = 'none',
task_type = "CAUSAL_LM",
lora_dropout = 0.1
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config, #quantize to 4 bit
    device_map = 'auto'
)
base_model.config.use_cache = False



tokenizer = LlamaTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


#tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(examples['text']), batched = True)
#tokenized_eval_dataset = test_dataset.map(lambda examples: tokenizer(examples['text']), batched = True)

arguments = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_ratio = 0.03,
    max_steps = 1000,
    learning_rate = 2e-4,
    fp16 = True,
    logging_steps = 50,
    output_dir = 'outputs',
    optim = optim,
    lr_scheduler_type='constant',
    evaluation_strategy='steps',
    eval_steps = 100,
    save_safetensors=True
)

trainer = SFTTrainer(
  model = base_model,
  train_dataset = train_dataset,
  eval_dataset = test_dataset,
  peft_config = peft_config,
  dataset_text_field = 'text',
  max_seq_length=1000,
  tokenizer = tokenizer,
  args = arguments,
  packing = False
)

trainer.train()
trainer.model.save_pretrained('alpaca_peft_model')
trainer.push_to_hub('done!')
