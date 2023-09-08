from fastapi import FastAPI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)


model_path ='./llamamodel'
tokenizer_path = './llamatokenizer'

model = AutoModelForCausalLM.from_pretrained(model_path)
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

