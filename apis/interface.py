import streamlit as st
import requests
import time
st.title('Chef Barks!😋')
headers = {'accept': 'application/json'}

url = 'http://0.0.0.0:4000/get_rag_response'

text = st.text_input('Type request: ')

if text:
    with st.spinner('Loading...'):
        params = {'text': text}
        response = requests.post(url, headers=headers, params=params)
        st.write(response.json())

