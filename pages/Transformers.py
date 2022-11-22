from transformers import pipeline
import streamlit as st
import pandas as pd

class TransformerModelClass():
	def __init__(self, task_type):
		st.write("model is loading ..., Please wait...")
		self.task_type = task_type
		self.classifier = self.get_model_obj(task_type)
		st.write("Model loaded!!!")

	@st.cache(allow_output_mutation =True)
	def get_model_obj(self, task_type):
		return pipeline(task_type)

	def get_predictions(self, text):
		if self.task_type == 'zero-shot-classification':
			return str(self.classifier(text, self.categories))
		return str(self.classifier(text))

@st.cache(allow_output_mutation =True, suppress_st_warning =True)
def input_data(data_file):
    ''' This function for taking two excel sheets as input'''
    st.write("Reading data from the file ....")
    try:
    	data_df = pd.read_excel(data_file)
    except ValueError:
    	st.write("Valuer error occured, Reading file as a csv")
    	data_df = pd.read_csv(data_file)
    return data_df

task_type = str(st.selectbox('Select Task_type from the list', tuple(["sentiment-analysis", "text-generation", "ner", "zero-shot-classification"])))
obj_main = TransformerModelClass(task_type)
data_type = st.selectbox("Apply model on ", tuple(["Single Sample", "DataSet"]))
if task_type == "zero-shot-classification":
	categories = str(st.text_input('Enter classes as comma separated values.....,'))
	obj_main.categories = list(categories.split(","))

if data_type == "Single Sample":
	text = st.text_input('Enter Text and hit enter:')
	if text:
		out = obj_main.get_predictions(text)
		st.write('Output:', out)

else:
	data_file = st.file_uploader("Choose a file data set from your system ...")
	data_df = input_data(data_file)
	st.write("Preview of Input dataset: ", data_df.head())
	text_col = str(st.selectbox('Select column containing text data to get predictions', tuple(data_df.columns)))
	data_df[task_type+'_on_'+text_col] = data_df[text_col].apply(obj_main.get_predictions)
	st.write("Preview of Output Dataset: ", data_df.head())

	with st.form("form3"):
		download_or_not = st.form_submit_button(label = 'Download Predictions')
	if download_or_not==True:
		try:
			data_df.to_excel("transformers_outs.xlsx")
			st.write("file downloaded with name transformers_outs.xlsx")
		except:
			data_df.to_csv("transformers_outs.csv")
			st.write("file downloaded with name transformers_outs.csv")






