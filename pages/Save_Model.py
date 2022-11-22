import streamlit as st
from joblib import dump, load
import json

def save_model(model, model_name):
	dump(model, str(model_name)+'.joblib')
	st.write("Model dumped") 

model = st.session_state["trained_model"]
file_name = st.text_input('Enter name of the model and hit enter:')
if len(file_name):
	save_model(model, file_name)

	json_object = json.dumps({"file_name": file_name}, indent = 4)
	with open("sample.json", "w") as outfile:
		outfile.write(json_object)

	with open("sample.json", "w") as outfile:
		st.write(outfile.read())




