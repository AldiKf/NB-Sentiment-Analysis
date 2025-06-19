import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

# --- Setup NLTK resource directory (local) ---
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR)

# --- Setup preprocessing tools ---
stop_words_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
tokenizer = RegexpTokenizer(r'\w+')

# --- Preprocessing Functions ---
def cleaningText(text):
    text = re.sub(r'[@#][A-Za-z0-9]+', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return tokenizer.tokenize(text)

def filteringText(tokens, stopwords_set):
    return [word for word in tokens if word not in stopwords_set]

def lemmatizeText(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def stemmingText(text):
    return stemmer.stem(text)

def toSentence(tokens):
    return ' '.join(tokens)

def preprocess(text, lang='english'):
    text = cleaningText(text)
    text = casefoldingText(text)
    if lang == 'english':
        tokens = tokenizingText(text)
        tokens = filteringText(tokens, stop_words_en)
        tokens = lemmatizeText(tokens)
        return toSentence(tokens)
    elif lang == 'indonesian':
        return stemmingText(text)
    return text

# --- Load Model & Vectorizer ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "naive_bayes_model_ros.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# --- Streamlit UI ---
st.title("📊 Review Sentiment Analysis App - Naive Bayes")

menu = st.sidebar.selectbox("Select Model", ["Sentence Prediction", "CSV File Prediction"])

if menu == "Sentence Prediction":
    st.subheader("Enter Sentence")
    user_input = st.text_area("Write your sentence here...")
    language = st.radio("Select Input Language", ["English", "Indonesian"])

    if st.button("Predict Your Sentences"):
        if user_input.strip() != "":
            if language == 'Indonesian':
                user_input = GoogleTranslator(source='auto', target='en').translate(user_input)
                processed = preprocess(user_input, lang='english')
            else:
                processed = preprocess(user_input, lang='english')
            vec = vectorizer.transform([processed])
            pred = model.predict(vec)[0]
            st.success(f"🎯 Prediction Results: **{pred}**")
        else:
            st.warning("Blank text!")

elif menu == "CSV File Prediction":
    st.subheader("📁 Upload CSV")
    st.markdown("**Attention!** The first row must contain column names!")
    st.markdown("- Line 1: `text_sentiment`")
    st.markdown("- Line 2: `This app is great!`")
    language = st.radio("Select Input Language", ["English", "Indonesian"])
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Select a column:", df.columns)

        if st.button("Prediction"):
            lang_code = 'english' if language == 'English' else 'indonesian'
            df['Preprocessed'] = df[text_col].astype(str).apply(lambda x: preprocess(x, lang=lang_code))
            df = df[df['Preprocessed'].str.lower() != 'nan']
            vecs = vectorizer.transform(df['Preprocessed'])
            df['Prediction'] = model.predict(vecs)
            st.dataframe(df[[text_col, 'Prediction']])

            # --- Bar Chart ---
            st.subheader("📊 Distribution of Prediction Results")
            pred_count = df['Prediction'].value_counts()

            fig, ax = plt.subplots()
            pred_count.plot(kind='bar', color=['#99FF99', '#FF9999', '#66B3FF'])
            plt.xlabel('Prediction Labels')
            plt.ylabel('Amount')
            plt.title('Distribution of Prediction Results')
            st.pyplot(fig)

            # --- Word Cloud ---
            st.subheader("☁️ WordCloud - Frequently Appearing Words")
            all_text = ' '.join(df['Preprocessed'])
            wc = WordCloud(width=800, height=400, background_color='white',
                        stopwords=STOPWORDS, colormap='viridis').generate(all_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)

            # --- Pie Chart ---
            st.subheader("📊 Percentage Distribution of Predicted Results")
            fig2, ax2 = plt.subplots()
            labels = pred_count.index
            sizes = pred_count.values
            colors = ['#99FF99', '#FF9999', '#66B3FF']

            wedges, texts, autotexts = ax2.pie(
                sizes,
                autopct='%1.1f%%',
                colors=colors,
                startangle=140,
                textprops=dict(color="black")
            )
            legend_labels = [f'{label}: {size}' for label, size in zip(labels, sizes)]
            ax2.legend(wedges, legend_labels, title="Number of Reviews", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            ax2.axis('equal')
            st.pyplot(fig2)

            # --- Download Hasil ---
            csv_result = df[[text_col, 'Prediction']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_result,
                file_name='prediction_results.csv',
                mime='text/csv'
            )
