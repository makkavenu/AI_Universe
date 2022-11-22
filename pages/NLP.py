import streamlit as st
import spacy
import pandas as pd 
import json
from spacy.matcher import Matcher

@st.cache(allow_output_mutation =True)
def load_spacy():
	nlp = spacy.load('en_core_web_sm')
	return nlp

nlp = load_spacy()
selection = st.selectbox("Select one of the operation: ", ["None","POS_Extractor", "POS Pattern Generator", "POS Pattern Matcher"])


class PosPatterGenerator():
	def __init__(self):
		pass

	def pos_detector(self, phrase_to_match, key_phrase = ""):
	    doc_nlp = nlp(phrase_to_match)
	    pos_lst = []
	    for token in doc_nlp:
	    	pos_lst.append(token.pos_)
	    return pos_lst

	def pos_pattern_generator(self, pos_lst):
	    pattern = []
	    for pos in pos_lst:
	        sub_pattern_dic = {"POS": pos, "OP":"*"}
	        pattern.append(sub_pattern_dic)
	    print("pattern generated is ",pattern )
	    return pattern

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
	def pattern_matcher(self, pattern, texts, size):
	    matcher = Matcher(vocab=nlp.vocab)
	    matcher.add("p1",[pattern])
	    phrases_lst = []
	    if size == "Single Sample":
	    	text = texts 
	    	doc = nlp(text)
	    	res = matcher(doc, as_spans=True)
	    	res = self.ent_ext(res)
	    	return res
	    if size == "df":
		    for text in texts:
		        doc = nlp(text)
		        res = matcher(doc,as_spans=True)
		        res = self.ent_ext(res)
		        res = [phrase.lower() for phrase in res]
		        phrases_lst.extend(res)
		    return phrases_lst
if selection == "POS_Extractor":
	text = st.text_input("Enter text to get pos , dep for each word in text: ")
	st.write("Ex: Apple is looking at buying U.K. startup for $1 billion")
	if text:
		doc = nlp(text)

		words = []
		pos_lst = []
		dep_lst = []
		for token in doc:
			words.append(token.text)
			pos_lst.append(token.pos_)
			dep_lst.append(token.dep_)
		res_df = pd.DataFrame({"Tokens": words, "POS":pos_lst, "DEP":dep_lst})
		st.write("OUTPUT Preview 10:", res_df.head(10))
		with st.form("form4"):
			download_or_not = st.form_submit_button(label = 'Download Output')
		if download_or_not==True:
			res_df.to_excel("spacy_pos_outs.xlsx")
			st.write("file downloaded with name spacy_pos_out.xlsx in current directory")

elif selection == "POS Pattern Generator":
	obj = PosPatterGenerator()
	text = st.text_input("Enter Phrase to which you wants generate pos pattern: ")
	st.write("Ex: playing cricket")
	if text:
		pos_lst = obj.pos_detector(text)
		pattern = obj.pos_pattern_generator(pos_lst)
		st.write("Output Pattern: ", pattern)
		st.write("NOTE: Use the above generated pattern to match phrases by selecting 'POS Pattern Matcher' option in above dropdown.....")

elif selection == "POS Pattern Matcher":
	obj = PosPatterGenerator()
	pat_text = st.text_input("Enter POS Pattern: ")
	pat_text = pat_text.replace("'", '"')
	pattern = json.loads(pat_text)
	#selection = st.selectbox("Select input data type: ", ["None","Single Sample", "df"])
	text = st.text_input("Enter text to match above POS pattern: ")
	st.write("EX: playing football")
	if text:
		res = obj.pattern_matcher(pattern, text, "Single Sample")
		st.write("OUTPUT: ", res)








