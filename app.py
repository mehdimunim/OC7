from flask import Flask, request, jsonify
import pickle
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

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





app = Flask(__name__)

# Charger le modèle et les objets de prétraitement
model = pickle.load(open('models/model_lr.pkl', 'rb'))  # Charger le modèle
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))  # Charger le vectorizer TF-IDF
selector = pickle.load(open('models/feature_selector_tfidf.pkl', 'rb'))  # Charger le sélecteur de features pour TF-IDF

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le texte du tweet depuis la requête
    tweet = request.form.get('tweet')

    # Prétraiter le tweet
    tweet_cleaned = clean_tweet(tweet)

    # Vectoriser le tweet avec TF-IDF
    tweet_vec = vectorizer.transform([tweet_cleaned])

    # Appliquer la sélection de features
    tweet_vec = selector.transform(tweet_vec)

    # Prédire le sentiment
    sentiment = model.predict(tweet_vec)[0]

    # Convertir le sentiment en "positif" ou "négatif"
    if sentiment != 0:  # Assuming 0 is negative and 4 is positive
        sentiment_label = "positif"
    else:
        sentiment_label = "négatif"

    # Retourner le sentiment prédit
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
