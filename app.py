from flask import Flask, request, jsonify
import pickle
import sys
import numpy as np
from gensim.models import Doc2Vec  # Import de Doc2Vec

# Ajouter le chemin du dossier src pour importer les modules
sys.path.append('src')
import preprocessing

app = Flask(__name__)

# Charger le modèle et les objets de prétraitement
model = pickle.load(open('models/model_lr.pkl', 'rb'))  # Charger le modèle
doc2vec_model = Doc2Vec.load('models/doc2vec_model.pkl')  # Charger le modèle Doc2Vec
selector = pickle.load(open('models/feature_selector_doc2vec.pkl', 'rb'))  # Charger le sélecteur de features pour Doc2Vec

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le texte du tweet depuis la requête
    tweet = request.form.get('tweet')

    # Prétraiter le tweet
    tweet_cleaned = preprocessing.clean_tweet(tweet)

    # Vectoriser le tweet avec Doc2Vec
    tweet_vec = doc2vec_model.infer_vector(tweet_cleaned.split())
    tweet_vec = np.array([tweet_vec])  # Convertir en array numpy

    # Appliquer la sélection de features
    tweet_vec = selector.transform(tweet_vec)

    # Prédire le sentiment
    sentiment = model.predict(tweet_vec)[0]

    # Convertir le sentiment en "positif" ou "négatif"
    if sentiment != 0:
        sentiment_label = "positif"
    else:
        sentiment_label = "négatif"

    # Retourner le sentiment prédit
    return jsonify({'sentiment': sentiment_label})

if __name__ == '__main__':
    app.run(debug=True)
