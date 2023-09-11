#!/usr/bin/bash

pip install -r requirements.txt
pip install -q git+https://github.com/huggingface/peft.git git+https://github.com/huggingface/transformers.git
streamlit run --server.address 0.0.0.0 interface.py
uvicorn main:app --reload --host 0.0.0.0 --port 4000

