import streamlit as st
st.set_page_config(
	page_title = "Upload Data Page")

st.write("Required packages importing, Please wait..........")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

st.write("Packages importing success")
data_file = st.file_uploader("Choose a file data set from your system ...")

st.sidebar.success("Select a page from above list.")


@st.cache(suppress_st_warning=True)
def input_data(data_file):
    ''' This function for taking two excel sheets as input'''
    st.write("Reading data from the file ....")
    try:
    	data_df = pd.read_excel(data_file)
    except ValueError:
    	st.write("Valuer error occured, Reading file as a csv")
    	data_df = pd.read_csv(data_file)
    return data_df

def get_train_and_test_arrays(data_df, dep_feature_cols, ind_feature_cols):
	'''This function is to select text column in input dataframe'''
	st.write("Inside get_train_and_test_arrays method")
	x = list(data_df[dep_feature_cols[0]])
	#st.write("dimesion of x is: ", x.shape)
	x = np.reshape(x,(-1,1))
	st.write("dimesion of x after reshaping is: ", x.shape)

	y = list(data_df[ind_feature_cols[0]])
	y = np.reshape(y,(-1,1))
	st.write("dimesion of x after reshaping is: ", y.shape)
	return x,y


try:
	data_df = input_data(data_file)
	if "data_df" not in st.session_state:
		st.session_state["data_df"] = data_df

except Exception as e:
	st.write("Thanks ")
	pass
	#st.write(e.msg)



