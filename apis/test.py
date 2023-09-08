from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
class Item(BaseModel):
    name:str
    description:str | None = None
    price:float
    tax: float | None = None

app = FastAPI()
pipe = pipeline('sentiment-analysis', model = './model/')
print('pipeline initialised')

@app.get("/items/{item_id}")
async def read_item(item_id):
    return {'message': "hello world", "itemId": item_id}


@app.get('/')
async def confirm():
    return {'message':'Everthing initialised'}


@app.post('/predict/')
async def predict(text):
    return pipe(text)[0]