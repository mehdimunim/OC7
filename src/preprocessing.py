from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer 
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import pickle

def vectorize_tfidf(X_train, X_test):
    """
    Vectorise les données textuelles avec TF-IDF et sauvegarde le vectorizer.

    Args:
        X_train : Les données d'entraînement.
        X_test : Les données de test.

    Returns:
        X_train_tfidf, X_test_tfidf : Les données vectorisées avec TF-IDF.
    """

    vectorizer_tfidf = TfidfVectorizer()
    X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
    X_test_tfidf = vectorizer_tfidf.transform(X_test)

    # Sauvegarder le vectorizer TF-IDF
    with open('../models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer_tfidf, f)

    return X_train_tfidf, X_test_tfidf


def vectorize_word2vec(X_train, X_test):
    """
    Vectorise les données textuelles avec Word2Vec et sauvegarde le modèle.

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

    # Sauvegarder le modèle Word2Vec
    model_w2v.save('../models/word2vec_model.pkl')

    return X_train_w2v, X_test_w2v


def vectorize_doc2vec(X_train, X_test):
    """
    Vectorise les données textuelles avec Doc2Vec et sauvegarde le modèle.

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

    # Sauvegarder le modèle Doc2Vec
    model_d2v.save('../models/doc2vec_model.pkl')

    return X_train_d2v, X_test_d2v

def reduce_dimensionality(X_train, X_test, y_train, k=1000, nom="tfidf"):
    """
    Réduit la dimensionnalité des données en utilisant SelectKBest avec le test ANOVA F-value
    et sauvegarde le sélecteur de features.

    Args:
        X_train : Les données d'entraînement.
        X_test : Les données de test.
        y_train : Les étiquettes d'entraînement.
        k : Le nombre de features à sélectionner.
        nom : Le nom de la méthode de vectorisation (tfidf, word2vec, doc2vec, etc.).

    Returns:
        X_train_reduced, X_test_reduced : Les données avec une dimensionnalité réduite.
    """

    selector = SelectKBest(f_classif, k=k)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)

    # Sauvegarder le sélecteur de features avec un nom spécifique à la méthode
    with open(f'../models/feature_selector_{nom}.pkl', 'wb') as f:
        pickle.dump(selector, f)

    return X_train_reduced, X_test_reduced
