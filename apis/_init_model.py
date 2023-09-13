from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)
import torch
import weaviate
from sentence_transformers import SentenceTransformer
from prompts import SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE, RETRIEVER_PROMPT_TEMPLATE, OPENAI_SYSTEM_PROMPT_TEMPATE, OPENAI_USER_TEMPLATE

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import openai
import getpass
# Load the Lora model


# we could for starters tell the user to b as detailed with their request as they can
class Models:
    def __init__(self):
        peft_model_id = "odunola/bloomz_reriever_instruct"
        config = PeftConfig.from_pretrained(peft_model_id)
        rephrase_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit = True, device_map = 'auto')
        self.tokenizer2 = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        self.llm2 = PeftModel.from_pretrained(rephrase_model, peft_model_id)
        openai_password = getpass.getpass("Enter your openai api key: ")
        openai.api_key = openai_password
        self.semantic_model = SentenceTransformer('thenlper/gte-large')
        self.client = weaviate.Client(
                url="https://testingserver-8otaf3tj.weaviate.network", #for testing
            )
    def _retrieve_from_db(self, text): #this gets the context from the vector db
        query = f"To generate a representation for this sentence for use in retrieving related articles: {text}"
        query_vector = self.semantic_model.encode(query)
        response = self.client.query.get(
        "Recipes",
        ["texts"]
            ).with_limit(3).with_near_vector(
                {'vector': query_vector}
            ).do()
        return text, response['data']['Get']['Recipes']

    def return_recipe(self, text):
        interpretation = self._generate_interpretation(text)
        context = self._retrieve_from_db(interpretation)
        user_prompt = OPENAI_USER_TEMPLATE.format(context = context,request = interpretation)
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system", "content":OPENAI_SYSTEM_PROMPT_TEMPATE},{"role": "user", "content": user_prompt}])
        response = chat_completion['choices'][0]['message']['content']
        return response
    def predict(self, text):
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system", "content":SIMPLE_PREDICTION_OPENAI_PROMPT_TEMPLATE},{"role": "user", "content": text}])
        response = chat_completion['choices'][0]['message']['content']
        return response
    def _generate_interpretation(self,text):
            prompt = RETRIEVER_PROMPT_TEMPLATE.format(request = text)
            tokens = self.tokenizer2(prompt, return_tensors = 'pt')
            outputs =  self.llm2.generate(input_ids = tokens['input_ids'].to('cuda'), temperature = 0, max_length = 300)
            response = self.tokenizer2.decode(outputs[0], skip_special_tokens=True).split('Interpretation:')[-1]       
            return response
            

