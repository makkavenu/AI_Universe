import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MlModel():
	def __init__(self):
		self.dep_fea_values_lst = []
		pass

	def dep_features_processing(self, dep_df):
		cat_cols = self.get_category_cols(dep_df)
		dep_df = self.convert_category_cols_to_numeric(dep_df, cat_cols)
		st.write("Final dep_feature_df is: ", dep_df.head())
		return dep_df
		
	def ind_features_processing(self, ind_df):
		if len(list(ind_df._get_numeric_data().columns))==1:
			pass
		else:
			col = list(ind_df.columns)[0]
			ord_enc = OrdinalEncoder()
			ind_df[col] = ord_enc.fit_transform(ind_df[[col]])
			self.dep_fea_values_lst = list(list(ord_enc.categories_)[0])
			st.write("index to cat_value map is:")

		st.write("final ind_df is: ", ind_df.head())
		return ind_df

	def get_category_cols(self, data_df):
		numerical_cols = data_df._get_numeric_data().columns
		category_cols = list(set(data_df.columns)-set(numerical_cols))
		return category_cols

	def convert_category_cols_to_numeric(self, feature_df, cat_cols):
		for cat_col in cat_cols:
			num_df = pd.get_dummies(feature_df[cat_col], prefix=str(cat_col))
			#st.write("num_df is: ", num_df.head())
			feature_df = feature_df.drop(cat_col, axis =1)
			feature_df = pd.concat([feature_df, num_df], axis = 1)
		return feature_df

	def append_dep_indep_dfs(self, dep_df, ind_df):
		final_df = pd.concat([dep_df, ind_df], axis=1)



	def get_train_and_test_arrays(self, data_df, dep_feature_cols, ind_feature_cols):
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



class LinearRegressionClass(MlModel):
	#Same for linear regresion and multiple linear regression
	def __init__(self):
		super().__init__()
		pass
	def get_model_obj(self):
		return LinearRegression()
	def validate_model(self, model, x_test, y_test):
		st.write("Testing on test data ...")
		y_pred = model.predict(x_test)
		score = mean_squared_error(y_test, y_pred)
		st.write('MSE:', score)

	def save_model(self, model, model_name):
		dump(model, str(model_name)+'.joblib')
		st.write("Model dumped") 
		pass

class ClassificationTasks(MlModel):
	def __init__(self):
		super().__init__()
	def validate_model(self, model, x_test, y_test):
		st.write("Testing on test data ...")
		y_pred = model.predict(x_test)
		matrix = confusion_matrix(y_test, y_pred) #labels = self.dep_fea_values_lst)
		# disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels= self.dep_fea_values_lst)
		st.write("Confusion matrix is :", matrix)
		st.write(self.dep_fea_values_lst)

		st.write("accuracy score: ",accuracy_score(y_test, y_pred))
		st.write("precision score: ",precision_score(y_test, y_pred))
		st.write("recall score: ",recall_score(y_test, y_pred))
		st.write("f1-score : ",f1_score(y_test, y_pred))

		# disp.plot()
		# st.write(plt.show())
		return matrix

	def save_model(self, model, model_name):
		dump(model, str(model_name)+'.joblib')
		st.write("Model dumped") 
		pass




class LogisticRegressionClass(ClassificationTasks):
	def __init__(self):
		super().__init__()
		pass
	def get_model_obj(self):
		return LogisticRegression()


class SVMClass(ClassificationTasks):
	def __init__(self):
		super().__init__()
		pass
	def get_model_obj(self):
		return LinearSVC(random_state=0, tol=0.3,loss="squared_hinge",multi_class="crammer_singer")

class MLPClass(ClassificationTasks):
	def __init__(self):
		super().__init__()
		pass
	def get_model_obj(self):
		return MLPClassifier(activation='tanh',solver='sgd',hidden_layer_sizes=(80,50,30,8),random_state=4,alpha=0.1,batch_size=71)


model_name_to_model_obj = {"Linear Regression": LinearRegressionClass(), "Logistic Regression": LogisticRegressionClass(),
							"SVM": SVMClass(), "MLP": MLPClass()}
try:
	data_df = st.session_state["data_df"]


	st.write("Preview of dataset you selected: ",data_df.head())
	model_name = st.selectbox('Select model from below list', tuple(["Linear Regression", "Logistic Regression", "SVM", "MLP"]))
	columns = list(data_df.columns)
	dependent_features = st.multiselect('Select Dependent features in the data ...', columns)
	independent_features = st.multiselect('Select Independent features in the data ...', list(set(columns) - set(dependent_features)))

	dep_df = data_df[list(dependent_features)]
	ind_df = data_df[list(independent_features)]

	with st.form("my_form"):
		train_or_not = st.form_submit_button(label="Train Model", help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
	#st.write("train or not flag is: ", train_or_not)

	if train_or_not == True:
		model_class = model_name_to_model_obj[model_name]
		dep_df = model_class.dep_features_processing(dep_df)
		ind_df = model_class.ind_features_processing(ind_df)

		#x, y = model_class.get_train_and_test_arrays(data_df, dependent_features, independent_features)
		
		model = model_class.get_model_obj()

		x_train, x_test, y_train, y_test = train_test_split(dep_df, ind_df, test_size = 1/3, random_state = 0)
		st.write("type of x_train is:", type(x_train))
		st.write(x_train.shape, y_train.shape)
		st.write(x_test.shape, y_test.shape)
		st.write("Model is training ...")
		model.fit(x_train,y_train)
		st.write("Training over")
		model_class.validate_model(model, x_test, y_test)

		st.session_state["trained_model"] = model
		# st.write("Saving the model ...")
		# model_class.save_model(model, "sample1")

		# save_or_not = st.selectbox('Do you want to save the model No/Yes:', tuple(["No", "Yes"]))

		# if save_or_not == "Yes":
		# 	st.write("Saving the model ...")
		# 	model_class.save_model(model, "sample2")

		def sample_fun():
			st.write("Hi")

		with st.form("my_form_2"):
			save_or_not = st.form_submit_button(label = "Save Model", on_click = sample_fun) #st.form_submit_button(label="Save Model", help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
		# st.write(save_or_not)
		# if save_or_not == True:
		# 	st.write("Saving the model ...")
		# 	model_class.save_model(model, "sample2")
except:
	st.write("Make sure to upload file(xlsx/csv) in tab named 'ai_univers_code'")
