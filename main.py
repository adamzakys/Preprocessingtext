import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
from gensim.models import LdaModel
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

nltk.download("punkt")
nltk.download("stopwords")

# Define a function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[.,():-]', '', text)
    text = re.sub(r'\d+', '', text)
    # Additional preprocessing steps can be added here
    return text

# Define a function for stemming
def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Sidebar
st.sidebar.title("Clustering Options")
clustering_method = st.sidebar.selectbox("Pilih modeling yang akan di Clustering", ["LDA", "TF-IDF"])
n_clusters = st.sidebar.slider("Pilih jumlah Cluster", min_value=2, max_value=10, value=3)

# Upload and preprocess data
st.title("Preprocessing dan Clustering Modeling Topik Tugas akhir skripsi mahasiswa UTM dengan sumber pta.trunojoyo.ac.id")
st.subheader("Muhammad Adam Zaky Jiddyansah")
st.subheader("210411100234")

uploaded_file = st.file_uploader("Upload dokumen CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the original data
    st.subheader("Original Data")
    st.dataframe(df)

    # Preprocess the text data and create a 'processed_abstrak' column
    st.subheader("Preprocessing Steps:")
    st.write("1. Text Tokenization")

    df['processed_abstrak'] = df['Abstrak'].apply(lambda x: nltk.word_tokenize(x) if isinstance(x, str) else [])

    st.dataframe(df[['Judul', 'processed_abstrak']])

    st.write("2. Punctuation Removal")
    df['processed_abstrak'] = df['processed_abstrak'].apply(lambda tokens: [re.sub(r'[.,():-]', '', token) for token in tokens])
    
    st.dataframe(df[['Judul', 'processed_abstrak']])

    st.write("3. Stopword Removal")
    stop_words = set(stopwords.words("indonesian"))
    df['processed_abstrak'] = df['processed_abstrak'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])

    st.dataframe(df[['Judul', 'processed_abstrak']])

    # LDA section
    if clustering_method == "LDA":
        # Process and analyze using LDA
        st.title("LDA Clustering")

        # Check if 'processed_abstrak' column contains lists or strings
        if df['processed_abstrak'].apply(lambda x: isinstance(x, list)).all():
            documents = df['processed_abstrak']
        else:
            documents = df['processed_abstrak'].apply(lambda x: x.split())

        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        lda_model = LdaModel(corpus, num_topics=n_clusters, id2word=dictionary, passes=15)

        topic_word_proposals = lda_model.get_topics()
        topic_word_proposals_df = pd.DataFrame(topic_word_proposals, columns=[dictionary[i] for i in range(len(dictionary))])

        st.write("Topic-Word Proposals:")
        st.dataframe(topic_word_proposals_df)

        document_topic_proposals = [lda_model.get_document_topics(doc) for doc in corpus]

        document_topic_proposals_df = pd.DataFrame(columns=["Judul"] + [f"Topic {i+1}" for i in range(lda_model.num_topics)])

        for i, doc_topic_proposals in enumerate(document_topic_proposals):
            row_data = {"Judul": df['Judul'].iloc[i]}
            for topic, prop in doc_topic_proposals:
                row_data[f"Topic {topic + 1}"] = prop
            document_topic_proposals_df = pd.concat([document_topic_proposals_df, pd.DataFrame([row_data])], ignore_index=True)

        document_topic_proposals_df = document_topic_proposals_df.fillna(0)

        st.write("Document-Topic Proposals:")
        st.dataframe(document_topic_proposals_df)

        kmeans_lda = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans_lda.fit(document_topic_proposals_df.iloc[:, 1:])
        silhouette_lda = silhouette_score(document_topic_proposals_df.iloc[:, 1:], kmeans_lda.labels_)

        document_topic_proposals_df['Cluster_LDA'] = kmeans_lda.labels_

        st.write("Cluster Results:")
        st.dataframe(document_topic_proposals_df[['Judul', 'Cluster_LDA']])
        st.write(f"Silhouette Score for LDA: {silhouette_lda}")

    elif clustering_method == "TF-IDF":
        # Process and analyze using TF-IDF with stemming
        st.title("TF-IDF Clustering with Stemming")
        documents_tfidf = df['Abstrak']  # Use 'Abstrak' column for TF-IDF

        # Apply stemming
        stemmed_documents = [stem_text(text) for text in documents_tfidf]
        documents_tfidf = stemmed_documents

        # Display the stemmed text
        st.subheader("Stemming Output:")
        st.write("Stemming is applied to the 'Abstrak' column:")
        st.dataframe(pd.DataFrame({'Abstrak': documents_tfidf, 'Stemmed Abstrak': stemmed_documents}))

        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tfidf_wm = tfidfvectorizer.fit_transform(documents_tfidf)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()

        df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
        df_tfidfvect.insert(0, 'Judul', df['Judul'])

        st.write("Stemmed TF-IDF Vectorizer:")
        st.dataframe(df_tfidfvect)

        kmeans_tfidf = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans_tfidf.fit(tfidf_wm)
        silhouette_tfidf = silhouette_score(tfidf_wm, kmeans_tfidf.labels_)

        df_tfidfvect['Cluster_TFIDF'] = kmeans_tfidf.labels_

        st.write("Cluster Results:")
        st.dataframe(df_tfidfvect[['Judul', 'Cluster_TFIDF']])
        st.write(f"Silhouette Score for TF-IDF: {silhouette_tfidf}")

    
