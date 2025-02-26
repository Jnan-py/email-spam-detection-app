import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_FILE = "spam_classifier.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
DATA_FILE = r"Dataset\spam.csv"

st.set_page_config(page_title="Email Spam Detection App", page_icon="📩", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE, encoding='latin-1')
    data.columns = ['Category', 'Message']
    data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    return data

data = load_data()

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        with open(MODEL_FILE, 'rb') as f:
            clf = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Spam'], test_size=0.25, random_state=42)
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        clf = MultinomialNB()
        clf.fit(X_train_vec, y_train)
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(clf, f)
        with open(VECTORIZER_FILE, 'wb') as f:
            pickle.dump(vectorizer, f)
    return clf, vectorizer

clf, vectorizer = load_or_train_model()

st.title("📩 Email Spam Detection App")
tabs = st.tabs(["📧 Spam Prediction", "📊 Data Visualization"])

with tabs[0]:
    st.header("🔍 Enter Email Content for Prediction")
    user_input = st.text_area("Type your email content below:")
    if st.button("Predict"):
        if user_input:
            input_vectorized = vectorizer.transform([user_input])
            prediction = clf.predict(input_vectorized)[0]
            result = "🚨 Spam!" if prediction == 1 else "✅ Not Spam (Ham)"
            st.subheader(f"Prediction: {result}")
        else:
            st.warning("Please enter some text to classify.")


with tabs[1]:
    st.header("📊 Dataset Overview and Visualizations")
    st.subheader("📌 Dataset Preview")
    n = st.slider("Select Number of Rows to Display", 1, 100, 5)
    st.dataframe(data.head(n))

    st.subheader("📊 Spam vs Ham Distribution")
    fig_pie = px.pie(data, names='Category', title='Spam vs Ham Distribution', hole=0.4)
    st.plotly_chart(fig_pie)
    
    st.subheader("📊 Message Length Distribution")
    data['Message Length'] = data['Message'].apply(len)
    fig_hist = px.histogram(data, x='Message Length', color='Category', nbins=50, title='Message Length Distribution')
    st.plotly_chart(fig_hist)
    
    st.subheader("☁️ Word Cloud for Messages")
    col1, col2 = st.columns(2)

    spam_words = ' '.join(data[data['Spam'] == 1]['Message'])
    spam_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(spam_words)
    fig, ax = plt.subplots()

    with col1:
        ax.imshow(spam_wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("Spam Word Cloud")
        st.pyplot(fig)
    
    ham_words = ' '.join(data[data['Spam'] == 0]['Message'])
    ham_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(ham_words)
    fig, ax = plt.subplots()

    with col2:
        ax.imshow(ham_wordcloud, interpolation='bilinear')
        ax.axis("off")
        ax.set_title("Ham Word Cloud")
        st.pyplot(fig)

    st.subheader("📊 Spam vs Ham Bar Plot")
    fig_bar = px.bar(data['Category'].value_counts(),
                     x=data['Category'].unique(),
                     y=data['Category'].value_counts(),
                     title='Spam vs Ham Count', labels={'x': 'Category', 'y': 'Count'})
    st.plotly_chart(fig_bar)


