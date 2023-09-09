from fastapi import FastAPI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)
import torch
from sentence_transformers import SentenceTransformer
import weaviate

client = weaviate.Client(
      url="https://testingserver-8otaf3tj.weaviate.network",
      additional_headers={
          "X-HuggingFace-Api-Key": "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"
      }
  )

model_path ='meta-llama/Llama-2-7b-chat-hf'
tokenizer_path = 'meta-llama/Llama-2-7b-chat-hf'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_dtype= 'nf4',
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant=False
)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config = bnb_config,
    device_map = 'auto',
    
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


retriever_model = SentenceTransformer('BAAI/bge-large-en')

app = FastAPI()
llm_template = """
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
    prompt = llm_template.format(question = text)
    tokens = tokenizer(prompt, return_tensors = 'pt')
    outputs = llm_model.generate(**tokens, max_length = 500)[0]
    response = tokenizer.decode(outputs, skip_special_tokens=True).split('[/INST]')[-1]
    return response

@app.post('/retrievedb')
def retrieve_from_vector_db(request):
    query = f"To generate a representation for this sentence for use in retrieving related articles: {request}"
    query_vector = retriever_model.encode(query)
    response = client.query.get(
    "Recipes",
    ["texts"]
        ).with_limit(3).with_near_vector(
            {'vector': query_vector}
        ).do()
    return response['data']['Get']['Recipes']





