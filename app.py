import streamlit as st
import pickle
import dill

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

ps = PorterStemmer()


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
transform_text = dill.load(open("transform_text.pkl", "rb"))

st.title("Email/SMS Spam Classifier!")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)


    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
