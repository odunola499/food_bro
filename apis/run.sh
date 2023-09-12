#!/usr/bin/bash

pip install -r requirements.txt
pip install -q git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git

uvicorn main:app --reload --host 0.0.0.0 --port 4000

