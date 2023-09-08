from fastapi import FastAPI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)


model_path ='meta-llama/Llama-2-7b-chat-hf'
tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_quant_type= 'nf4',
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config = bnb_config,
    device_map = 'auto',
    
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

app = FastAPI()
template = """
[INST] <<SYS>>
You are an helpful and smart assistant who knows his way around the kitchen and answers User's question as best as you can
<</SYS>>

User: {question} 
Assistant: [/INST]
"""

@app.get('/')
def check_load():
    return {'message': 'no errors found!'}

@app.post('/predict')
def predict(text):
    prompt = template.format(question = text)
    tokens = tokenizer(prompt, return_tensors = 'pt')
    output = model.generate(**tokens, max_length = 500).split('[/INST]')[-1]
    return output

