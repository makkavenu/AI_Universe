import streamlit as st
import spacy
import pandas as pd 
import json
from spacy.matcher import Matcher
import yake

@st.cache(allow_output_mutation =True)
def load_spacy():
	nlp = spacy.load('en_core_web_sm')
	return nlp

class ExtractPOS():
	def __init__(self):
		self.nlp = load_spacy()
		self.matcher = Matcher(self.nlp.vocab)
		self.kw_extractor = yake.KeywordExtractor(top=10, stopwords=None)

	def extract_insight(self, pattern, input_texts, size):

		if pattern == ["yake"]:
			st.write("inside yake in insight method")
			if size == "Single Sample":
				text = input_texts
				keywords = self.kw_extractor.extract_keywords(text)
				keys = []
				probs = []
				for (key, prob) in keywords:
				    keys.append(key)
				    probs.append(prob)
				# keys = self.ent_ext(keys)
				return keys
			else:
				texts = input_texts
				res_lst = []
				for text in texts:
					keywords = self.kw_extractor.extract_keywords(text)
					keys = []
					probs = []
					for (key, prob) in keywords:
					    keys.append(key)
					    probs.append(prob)

					# keys = self.ent_ext(keys)
					res_lst.append("|".join(keys))
				return res_lst
		#Below if else won't execute if yake selected beacause we are returning above
		if size=="Single Sample":
			self.matcher.add("p1", pattern)
			text = input_texts
			doc = self.nlp(text)
			res = self.matcher(doc, as_spans=True)
			res = self.ent_ext(res)
			return res
			# pattern = [{"POS": "NOUN", "OP":"*"}]
		else:
			self.matcher.add("p1", pattern)
			texts = input_texts
			res_lst = []
			for text in texts:
				doc = self.nlp(text)
				res = self.matcher(doc, as_spans=True)
				res = self.ent_ext(res)
				res = [phrase.lower() for phrase in res]
				res_lst.append(res)
			return res_lst

	def gram_cleaner(self, gramsubstring,gramstring):
		red_gramsubstring = []
		for m in range(len(gramsubstring)):
		    for n in range(len(gramstring)):
		        if gramsubstring[m] in gramstring[n]:
		            red_gramsubstring.append(gramsubstring[m])
		            break
		        else:
		            continue
		return list(set(gramsubstring) ^ set(red_gramsubstring))

	def ent_ext(self, matches):
		unigram = []
		bigram = []
		trigram = []
		quadgram = []
		pentgram = []
		hexagram = []
		heptagram = []
		octagram = []
		for span in matches:
		    #span = doc[start:end]
		    if len(span)==1:
		        unigram.append(span.text)
		    elif len(span)==2:
		        bigram.append(span.text)
		    elif len(span)==3:
		        trigram.append(span.text)
		    elif len(span)==4:
		        quadgram.append(span.text)
		    elif len(span)==5:
		        pentgram.append(span.text)
		    elif len(span)==6:
		        hexagram.append(span.text)
		    elif len(span)==7:
		        heptagram.append(span.text)
		    elif len(span)==8:
		        octagram.append(span.text)
		unigram = list(set(unigram))
		bigram = list(set(bigram))
		trigram = list(set(trigram))
		quadgram = list(set(quadgram))
		pentgram = list(set(pentgram))
		hexagram = list(set(hexagram))
		heptagram = list(set(heptagram))
		octagram = list(set(octagram))
		entities_list=[]
		entities_list.append(self.gram_cleaner(unigram,bigram))
		entities_list.append(self.gram_cleaner(bigram,trigram))
		entities_list.append(self.gram_cleaner(trigram,quadgram))
		entities_list.append(self.gram_cleaner(quadgram,pentgram))
		entities_list.append(self.gram_cleaner(pentgram,hexagram))
		entities_list.append(self.gram_cleaner(hexagram,heptagram))
		entities_list.append(self.gram_cleaner(heptagram,octagram))
		entities_list.append(octagram)

		entities_list = [x for x in entities_list if x != []]

		l_upd = []
		def reemovNestings(l):
		    for i in l:
		        if type(i) == list:
		            reemovNestings(i)
		        else:
		            l_upd.append(i)
		    return l_upd
		entities_list = reemovNestings(entities_list)

		for i in range(0,len(entities_list)-1):
		    for j in range(i+1,len(entities_list)):
		        if entities_list[i] in entities_list[j]:
		            entities_list.remove(entities_list[i])
		            break
		return entities_list


def input_data(data_file):
    ''' This function for taking two excel sheets as input'''
    st.write("Reading data from the file ....")
    try:
    	data_df = pd.read_excel(data_file)
    except ValueError:
    	st.write("Valuer error occured, Reading file as a csv")
    	data_df = pd.read_csv(data_file)
    return data_df

selection = st.selectbox("Select one of the operation: ", ["None","Extract Keywords","Extract Nouns", "Extract Verbs", "Extract Adjactives", "Use POS Pattern to extract"])
extraction_obj = ExtractPOS()
selection_to_pattern_dic = {"Extract Nouns": [{"POS":"NOUN", "OP":"*"}, {"POS":"PROPN", "OP":"*"}], "Extract Verbs":[{"POS":"VERB", "OP":"*"}] , "Extract Adjactives": [{"POS":"ADJ", "OP":"*"}]}

# if selection == "Extract Keywords":

pattern = None
if selection in ["Extract Nouns", "Extract Verbs", "Extract Adjactives"]:
	pattern = selection_to_pattern_dic[selection]
elif selection== "Use POS Pattern to extract":
	pattern = None
	pattern = st.text_input("Enter POS pattern: ")
	st.write('Ex: [{"POS": "ADJ", "OP":"*"}, {"POS": "NOUN", "OP":"*"}]')
	if pattern:
		pattern = json.loads(pattern.replace('""', '"*"'))

elif selection == "Extract Keywords":
	pattern = "yake"
if pattern:
	size = st.selectbox("Select size of the Input ", ["Single Sample", "Excel/CSV file"])
	if size == "Single Sample":
		text = st.text_input("Enter Text to extract: ")
		st.write("Ex: India is a great place to live")
		if text:
			res = extraction_obj.extract_insight([pattern], text, "Single Sample")
			st.write("OUPUT: ", res)
	else:
		data_file = st.file_uploader("Choose a file dataset(xlsx/csv) from your system ...")
		if data_file:
			data_df = input_data(data_file)
			cols = list(data_df.columns)
			columns = ["None"]
			columns.extend(cols)
			select_col = st.selectbox("Select Column from you want to extract NOUNS ", columns)
			if select_col != "None":
				texts = data_df[select_col].tolist()
				data_df[str(selection)+"_Results"] = extraction_obj.extract_insight([pattern], texts, "dataframe")
				st.write("Preview of output: ", data_df.head(10))
				with st.form("form5"):
					download_or_not = st.form_submit_button(label = 'Download Output')
				if download_or_not==True:
					file = str(selection)+"_results"+".xlsx"
					data_df.to_excel(file)
					st.write("file downloaded with name '",file,"' in current directory")











