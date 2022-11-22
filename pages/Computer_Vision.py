import streamlit as st
import openai
openai.api_key = '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' #b161098mail
import time

@st.cache(allow_output_mutation =True)
def request_image_model(description):
    try:
	    response = openai.Image.create(
	    	prompt = description,
	    	n = 1,
	    	size = "1024x1024"
	    	)

    except:
        time.sleep(10)
        return request_openai(question)

    return response

description = str(st.text_input('Describe image you want to generate and hit Enter'))
st.write("Ex: A Rich man looking at poor hut")

if description:
	response = request_image_model(description)
	answer = response["data"][0]["url"]	
	st.write("RESPONSE: Click the below link to see the generated image...,")
	st.write(answer)