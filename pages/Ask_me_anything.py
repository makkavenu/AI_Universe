import streamlit as st
import openai
openai.api_key = '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' #b161098mail
import time

@st.cache(allow_output_mutation =True)
def request_model(question):
	# #question = if str(question[-1])!="?": question = str(question)+"?"
	# question =  question if str(question[-1])=="?" else str(question)+"?"
	# st.write(question)
    try:
	    response = openai.Completion.create(
	        model= "text-davinci-002", #text-curie-001", #"curie:ft-setuserv:curie-setuserv-version-2-2022-08-23-10-22-01",#"curie:ft-setuserv:setuserv-curie-3-2022-08-26-06-47-56",#"curie:ft-setuserv:custom-curie-topic-extractor-2022-08-19-06-31-12",
	        prompt= ["Question: "+str(question)],
	        stop = "",
	        temperature=0,
	        max_tokens=200,
	        top_p=1,
	        frequency_penalty=0,
	        presence_penalty=0)

    except:
        time.sleep(10)
        return request_openai(question)

    return response

question = str(st.text_input('Enter Your Question Here and hit Enter'))
st.write("Ex: Give me life motivation?")

if question:
	response = request_model(question)
	answer = response["choices"][0]['text']
	st.write("RESPONSE: ",answer)

