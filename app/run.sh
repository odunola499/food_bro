#!/bin/bash

export OPENAI_API_KEY="YOUR OPENAI KEY"

python3 build_index.py
pip install -q -r requirements.txt
streamlit run --browser.serverAddress 0.0.0.0  guardrails.py
