#!/usr/bin/bash

pip install -q -r requirements.txt
pip install -q git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git
huggingface-cli login
uvicorn main:app --reload --host 0.0.0.0 --port 4000