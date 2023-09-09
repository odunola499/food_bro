from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)
import torch
import weaviate
from sentence_transformers import SentenceTransformer
from prompts import SIMPLE_PREDICTION_PROMPT_TEMPLATE, RETRIEVER_PROMPT_TEMPLATE, RETURN_RECIPE_TEMPLATE


# we could for starters tell the user to b as detailed with their request as they can
class Models:
    def __init__(self):
        llm_model_path = 'meta-llama/Llama-2-13b-chat-hf'
        llm_model_path_2 = 'Open-Orca/LlongOrca-7B-16k'
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_dtype= 'nf4',
            bnb_4bit_compute_dtype = torch.float16,
            bnb_4bit_use_double_quant=False
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                quantization_config = bnb_config,
                device_map = 'auto'
            )
        self.llm_2 = AutoModelForCausalLM.from_pretrained(
                llm_model_path_2,
                quantization_config = bnb_config,
                device_map = 'auto'
            )
        self.semantic_model = SentenceTransformer('thenlper/gte-large')
        self.client = weaviate.Client(
                url="https://testingserver-8otaf3tj.weaviate.network", #for testing
            )
    def _retrieve_from_db(self, text): #this gets the context from the vector db
        query = f"To generate a representation for this sentence for use in retrieving related articles: {text}"
        query_vector = self.retriever_model.encode(query)
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
        prompt = RETURN_RECIPE_TEMPLATE.format(context = context, request = text)
        tokens = self.tokenizer(prompt, return_tensors = 'pt')
        outputs = self.llm.generate(**tokens, max_length = 500, temperature = 0.8)[0]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True).split('[/INST]')[-1]
        return response
    def predict(self, text):
        prompt = SIMPLE_PREDICTION_PROMPT_TEMPLATE.format(question = text)
        tokens = self.tokenizer(prompt, return_tensors = 'pt')
        outputs = self.llm.generate(**tokens, max_length = 500)[0]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True).split('[/INST]')[-1]
        return response
    def _generate_interpretation(self,text):
            prompt = RETRIEVER_PROMPT_TEMPLATE.format(request = text)
            tokens = self.tokenizer(prompt, return_tensors = 'pt')
            outputs =  self.llm_2.generate(tokens, temperature = 0, max_length = 100)
            response = self.tokenizer.decode(outputs, skip_special_tokens=True).split('[/INST]')[-1]       
            return response
            