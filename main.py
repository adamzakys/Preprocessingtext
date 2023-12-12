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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
enable_stemming = st.sidebar.checkbox("Aktifkan Stemming (TF-IDF)", value=False)

# Data selection options
data_size_option = st.sidebar.radio("Pilih berapa banyak data yang digunakan", ["Semua Data", "Sebagian Data"])
if data_size_option == "Sebagian Data":
    data_fraction = st.sidebar.slider("Fraksi Data yang Digunakan", min_value=0.1, max_value=1.0, value=1.0, step=0.1)

# Upload and preprocess data
st.title("Preprocessing dan Clustering Modeling Topik Tugas akhir skripsi mahasiswa UTM dengan sumber pta.trunojoyo.ac.id")
st.subheader("Muhammad Adam Zaky Jiddyansah")
st.subheader("210411100234")

uploaded_file = st.file_uploader("Upload dokumen CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data size selection
    if data_size_option == "Sebagian Data":
        data_fraction = int(len(df) * data_fraction)
        df = df.iloc[:data_fraction]  # Mengambil data dalam urutan sesuai indeks

    # Display the original data
    st.subheader("Original Data")
    st.dataframe(df)

    # Preprocess the text data and create an 'abstrak' column
    st.subheader("Preprocessing Steps:")
    st.write("1. Text Tokenization")

    df['abstrak'] = df['abstrak'].apply(lambda x: nltk.word_tokenize(x) if isinstance(x, str) else [])

    st.dataframe(df[['judul', 'abstrak']])

    st.write("2. Punctuation Removal")
    df['abstrak'] = df['abstrak'].apply(lambda tokens: [re.sub(r'[.,():-]', '', token) for token in tokens])

    st.dataframe(df[['judul', 'abstrak']])

    st.write("3. Stopword Removal")
    stop_words = set(stopwords.words("indonesian"))
    df['abstrak'] = df['abstrak'].apply(lambda tokens: [token for token in tokens if token.lower() not in stop_words])

    st.dataframe(df[['judul', 'abstrak']])

    # LDA section
    if clustering_method == "LDA":
        # Process and analyze using LDA
        st.title("LDA Clustering")

        # Check if 'abstrak' column contains lists or strings
        if df['abstrak'].apply(lambda x: isinstance(x, list)).all():
            documents = df['abstrak']
        else:
            documents = df['abstrak'].apply(lambda x: x.split())

        dictionary = corpora.Dictionary(documents)
        corpus = [dictionary.doc2bow(doc) for doc in documents]

        lda_model = LdaModel(corpus, num_topics=n_clusters, id2word=dictionary, passes=15)

        topic_word_proposals = lda_model.get_topics()
        topic_word_proposals_df = pd.DataFrame(topic_word_proposals, columns=[dictionary[i] for i in range(len(dictionary))])

        st.write("Topic-Word Proposals:")
        st.dataframe(topic_word_proposals_df)

        document_topic_proposals = [lda_model.get_document_topics(doc) for doc in corpus]

        document_topic_proposals_df = pd.DataFrame(columns=["judul"] + [f"Topic {i+1}" for i in range(lda_model.num_topics)])

        for i, doc_topic_proposals in enumerate(document_topic_proposals):
            row_data = {"judul": df['judul'].iloc[i]}
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
        st.dataframe(document_topic_proposals_df[['judul', 'Cluster_LDA']])
        st.write(f"Silhouette Score for LDA: {silhouette_lda}")

        # Classification
        st.title("Klasifikasi")
        X = document_topic_proposals_df.drop(['judul', 'Cluster_LDA'], axis=1)
        y = df['label-topic']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train a K-Nearest Neighbors (KNN) classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
        knn_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = knn_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of K-Nearest Neighbors Classifier: {accuracy:.2f}")


    elif clustering_method == "TF-IDF":
        # Process and analyze using TF-IDF
        st.title("TF-IDF Clustering")

        documents_tfidf = df['abstrak']  # Use 'abstrak' column for TF-IDF

        if enable_stemming:
            # Apply stemming
            if df['abstrak'].apply(lambda x: isinstance(x, list)).all():
                stemmed_documents = [' '.join(x) for x in df['abstrak']]
            else:
                stemmed_documents = [stem_text(text) for text in documents_tfidf]
            documents_tfidf = stemmed_documents
        else:
            documents_tfidf = df['abstrak']  # Use the original 'abstrak' column

        # Display the stemmed text if stemming is enabled
        if enable_stemming:
            st.subheader("Stemming Output:")
            st.write("Stemming is applied to the 'abstrak' column:")
            st.dataframe(pd.DataFrame({'abstrak': df['abstrak'], 'Stemmed abstrak': stemmed_documents}))

        tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        tfidf_wm = tfidfvectorizer.fit_transform(documents_tfidf)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()

        df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
        df_tfidfvect.insert(0, 'judul', df['judul'])

        st.write("TF-IDF Vectorizer:")
        st.dataframe(df_tfidfvect)

        kmeans_tfidf = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans_tfidf.fit(df_tfidfvect.iloc[:, 1:])
        silhouette_tfidf = silhouette_score(df_tfidfvect.iloc[:, 1:], kmeans_tfidf.labels_)

        df_tfidfvect['Cluster_TFIDF'] = kmeans_tfidf.labels_

        st.write("Cluster Results:")
        st.dataframe(df_tfidfvect[['judul', 'Cluster_TFIDF']])
        st.write(f"Silhouette Score for TF-IDF: {silhouette_tfidf}")

        # Classification
        st.title("Klasifikasi")
        X = df_tfidfvect.drop(['judul', 'Cluster_TFIDF'], axis=1)
        y = df['label-topic']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train a K-Nearest Neighbors (KNN) classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed
        knn_classifier.fit(X_train, y_train)

        # Make predictions
        y_pred = knn_classifier.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of K-Nearest Neighbors Classifier: {accuracy:.2f}")
