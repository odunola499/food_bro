#!/bin/bash

export OPENAI_API_KEY="YOUR OPENAI KEY"

python3 build_index.py
streamlit run --browser.serverAddress 0.0.0.0  guardrails.py
