import streamlit as st
from joblib import dump, load
import numpy as np

model_name = st.text_input('Enter name of the model and hit enter:')
if len(model_name):
	model_loaded = load(model_name+'.joblib')
	#st.write("model is: ", model_loaded)
	x = [17.9900,10.3800,122.8000]
	x = np.reshape(x,(1,-1))

	y = model_loaded.predict(x)
	st.write(y)