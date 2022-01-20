# import needed libraries
from pprint import pprint
from string import punctuation

import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams.update({'figure.figsize': (15, 5), 'figure.dpi': 120})
plt.style.use('fivethirtyeight')

import nltk

import gensim, models
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import RegexpTokenizer
nltk.download('punkt')
nltk.download('stopwords') 
stemmer = WordNetLemmatizer()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score
from statistics import mean

from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

import os, re, operator, math, warnings
warnings.filterwarnings('ignore')


# web app title
st.title('Topic predictor')

# unsupervised clustering (lda)
## load trained lda model
lda_model =  models.LdaModel.load('../models/lda_model')

# define upload file component to import csv file
uploaded_csv = st.file_uploader("Choose a file")

st.cache()
def plot_clusters(df_uploaded, lda, n_topics):
    if df_uploaded is not None:
        # transform csv into pd Dataframe
        df_article = pd.read_csv(df_uploaded)

    # define preprocessing function
    def clean_articles(df, text_col, lang):
        df_copy = df.copy()
        
        # drop rows with empty values
        df_copy.dropna(inplace=True)
        
        # lower text
        df_copy['preprocessed_' + text_col] = df_copy[text_col].str.lower()

        # spell check
        #spell = Speller('en')
        #df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: spell(row))

        # remove punctuations
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: re.sub("[^-9A-Za-z ]", "" , row))
        
        # filter out stop word
        stop_words = set(stopwords.words(lang))        
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: ' '.join([word for word in row.split() if (not word in stop_words)]))

        # remove words with less than 2 letters
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: re.sub(r'\b\w{1,3}\b', '', row))

        # remove extra white spaces
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: re.sub("s+","", row))

        # lemmatize words
        wordnet_lemmatizer = WordNetLemmatizer()
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: wordnet_lemmatizer.lemmatize(row))

        # stemming words
        porter_stemmer = PorterStemmer()
        df_copy['preprocessed_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: porter_stemmer.stem(row))
        
        # tokenize 
        tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
        df_copy['tokenized_' + text_col] = df_copy['preprocessed_' + text_col].apply(lambda row: tokenizer.tokenize(row))
        
        return df_copy['tokenized_ABSTRACT']

    # define cleaned text variable
    df_cleaned = clean_articles(df_article, 'ABSTRACT', 'english') 

    # build a dictionary where for each tweet, each word has its own id.
    article_dictionary = corpora.Dictionary(df_cleaned)

    # build the corpus i.e. vectors with the number of occurence of each word per tweet
    # article_corpus = [article_dictionary.doc2bow(article) for article in df]

    # plot topics with top 10 words
    top_words = [[word for word,_ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _,beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs  = gridspec.GridSpec(round(math.sqrt(n_topics))+1,round(math.sqrt(n_topics))+1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20,15))
    for i in range(n_topics):
        fig, ax = plt.subplot(gs[i])
        plt.barh(range(10), top_betas[i][:10], align='center',color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_words[i][:10])
        plt.title("Topic "+str(i))
        
    st.pyplot(fig)

    # compute coherence ccore
    coherence_model_lda = CoherenceModel(model=lda, texts=df_cleaned, dictionary=article_dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    st.header(coherence_lda)

    # return prediction
    # predictions= lda[article_corpus]
    # st.header(predictions)

plot_clusters(uploaded_csv, lda_model, 5)
