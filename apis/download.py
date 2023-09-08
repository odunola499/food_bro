from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)
import torch

model_name = 'meta-llama/Llama-2-7b-chat-hf' 

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype= 'nf4',
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config,
    device_map = 'auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.save_pretrained('/llamatokenizer')
model.save_pretrained('./llamamodel')
