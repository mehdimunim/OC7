import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np



def clean_tweet(tweet):
    """
    Nettoie le texte d'un tweet en supprimant les mentions, les hashtags,
    les liens, les caractères spéciaux, la ponctuation, et en convertissant
    le texte en minuscules.

    Args:
        tweet : Le texte du tweet à nettoyer.

    Returns:
        Le texte du tweet nettoyé.
    """

    # Supprimer les mentions, les hashtags et les liens
    tweet = re.sub(r"@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|http\S+", "", tweet)

    # Supprimer les caractères spéciaux et la ponctuation
    tweet = re.sub(r"[^a-zA-Z ]", "", tweet)

    # Convertir le texte en minuscules
    tweet = tweet.lower()

    # Supprimer les mots vides
    stop_words = set(stopwords.words('english'))
    tweet = " ".join([word for word in tweet.split() if word not in stop_words])

    # Lemmatiser les mots
    lemmatizer = WordNetLemmatizer()
    tweet = " ".join([lemmatizer.lemmatize(word) for word in tweet.split()])

    return tweet


def vectorize_tfidf(X_train, X_test):
    """
    Vectorise les données textuelles avec TF-IDF.

    Args:
        X_train : Les données d'entraînement.
        X_test : Les données de test.

    Returns:
        X_train_tfidf, X_test_tfidf : Les données vectorisées avec TF-IDF.
    """

    vectorizer_tfidf = TfidfVectorizer()
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
    X_test_tfidf = vectorizer_tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def vectorize_word2vec(X_train, X_test):
    """
    Vectorise les données textuelles avec Word2Vec.

    Args:
        X_train : Les données d'entraînement.
        X_test : Les données de test.

    Returns:
        X_train_w2v, X_test_w2v : Les données vectorisées avec Word2Vec.
    """

    # Entraînement du modèle Word2Vec sur les tweets prétraités
    sentences = [tweet.split() for tweet in X_train]
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Fonction pour vectoriser un tweet en utilisant la moyenne des embeddings Word2Vec
    def vectorize_tweet_w2v(tweet):
        vectors = [model_w2v.wv[word] for word in tweet.split() if word in model_w2v.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model_w2v.vector_size)

    # Vectorisation des données d'entraînement et de test
    X_train_w2v = np.array([vectorize_tweet_w2v(tweet) for tweet in X_train])
    X_test_w2v = np.array([vectorize_tweet_w2v(tweet) for tweet in X_test])

    return X_train_w2v, X_test_w2v


def vectorize_doc2vec(X_train, X_test):
    """
    Vectorise les données textuelles avec Doc2Vec.

    Args:
        X_train : Les données d'entraînement.
        X_test : Les données de test.

    Returns:
        X_train_d2v, X_test_d2v : Les données vectorisées avec Doc2Vec.
    """

    # Préparation des données pour Doc2Vec
    documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(X_train)]

    # Entraînement du modèle Doc2Vec
    model_d2v = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4)

    # Fonction pour vectoriser un tweet en utilisant l'embedding Doc2Vec
    def vectorize_tweet_d2v(tweet):
        return model_d2v.infer_vector(tweet.split())

    # Vectorisation des données d'entraînement et de test
    X_train_d2v = np.array([vectorize_tweet_d2v(tweet) for tweet in X_train])
    X_test_d2v = np.array([vectorize_tweet_d2v(tweet) for tweet in X_test])

    return X_train_d2v, X_test_d2v