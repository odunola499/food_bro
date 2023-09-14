from fastapi import FastAPI
from prompts import YAML_CONTENT, RAG_COLANG_CONTENT
from _init_model import Models
from nemoguardrails import LLMRails, RailsConfig

model = Models()
config = RailsConfig.from_content(
    colang_content=RAG_COLANG_CONTENT,
    yaml_content=YAML_CONTENT
)
rag_rails = LLMRails(config)

rag_rails.register_action(action = model.generate_interpretation, name= "generate_interpretation")
rag_rails.register_action(action = model.retrieve, name = "retrieve")
rag_rails.register_action(action = model.reply, name = "rag")
app= FastAPI()


@app.get('/')
def check_load():
    return {'message': 'no errors found!'}

@app.post('/predict')
def predict(text):
    return model.predict(text)

@app.post('/retrievedb')
def retrieve_from_vector_db(request):
    return model.retrieve(request)

@app.post('/get_rag_response')
async def get_rag_response(text):
    result = await rag_rails.generate_async(prompt = text)
    return result

@app.post('/generate_representation')
def generate_representation(text):
    return model.generate_interpretation(text)



