from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)
import torch
import weaviate
from sentence_transformers import SentenceTransformer
# we could for starters tell the user to b as detailed with their request as they can
class Models:
    def __init__(self):
        llm_model_path = 'meta-llama/Llama-2-13b-chat-hf'
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
            ).with_limit(5).with_near_vector(
                {'vector': query_vector}
            ).do()
        return text, response['data']['Get']['Recipes']

    def return_recipe(self, text):
        context = self._retrieve_from_db(text)
        template = """You are a seasoned chef with a wealth of culinary expertise. You're known for your ability to craft exquisite dishes tailored to individual preferences. Request: Imagine a hungry visitor approaches you. They either express a craving for a specific dish, seek your culinary wisdom for a recommendation, or share a personal story that influences their food desires. In response, provide them with a detailed food recipe and step-by-step cooking instructions that precisely address their request, reflecting your culinary mastery.Use the context to guide your reasoning as it will suggest texts to help you in the request.
            Request: {request}
            Context: {context}
            Response: 
            """
        prompt = template.format(context = context, request = text)

    def predict(self, text):
        llm_template = """
            [INST] <<SYS>>
            You are an helpful and smart assistant who knows his way around the kitchen and answers User's question as best as you can
            <</SYS>>

            User: {question} 
            Assistant: [/INST]
            """
        prompt = llm_template.format(question = text)
        tokens = self.tokenizer(prompt, return_tensors = 'pt')
        outputs = self.llm.generate(**tokens, max_length = 500)[0]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True).split('[/INST]')[-1]
        return response
            