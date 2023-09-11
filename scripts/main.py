from fastapi import FastAPI

from _init_model import Models


model = Models()

app = FastAPI()


@app.get('/')
def check_load():
    return {'message': 'no errors found!'}

@app.post('/predict')
def predict(text):
    return model.predict(text)

@app.post('/retrievedb')
def retrieve_from_vector_db(request):
    return model._retrieve_from_db(request)

@app.post('/get_recipe')
def get_recipe(text):
    return model.return_recipe(text)

@app.post('/generate_representation')
def generate_representation(text):
    return model._generate_interpretation(text)


