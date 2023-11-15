#!/bin/bash


pip install -q -r requirements.txt

python3 build_index.py

streamlit run --browser.serverAddress 0.0.0.0  guardrails.py
