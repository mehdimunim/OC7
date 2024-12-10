from flask import Flask, request, jsonify
import pickle
import sys
import numpy as np
from gensim.models import Word2Vec # Import de Word2Vec

# Ajouter le chemin du dossier src pour importer les modules
sys.path.append('src')
import preprocessing

app = Flask(__name__)

# Charger le modèle et les objets de prétraitement
model = pickle.load(open('models/mon_modele_lr.pkl', 'rb'))  # Charger le modèle
word2vec_model = Word2Vec.load('models/word2vec_model.pkl')  # Charger le modèle Word2Vec
selector = pickle.load(open('models/feature_selector_word2vec.pkl', 'rb'))  # Charger le sélecteur de features pour Word2Vec

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer le texte du tweet depuis la requête
    tweet = request.form.get('tweet')

    # Prétraiter le tweet
    tweet_cleaned = preprocessing.clean_tweet(tweet)

    # Vectoriser le tweet avec Word2Vec
    def vectorize_tweet_w2v(tweet):
        vectors = [word2vec_model.wv[word] for word in tweet.split() if word in word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)
    tweet_vec = vectorize_tweet_w2v(tweet_cleaned)
    tweet_vec = np.array([tweet_vec])  # Convertir en array numpy pour la compatibilité avec SelectKBest

    # Appliquer la sélection de features
    tweet_vec = selector.transform(tweet_vec)

    # Prédire le sentiment
    sentiment = model.predict(tweet_vec)[0]

    # Retourner le sentiment prédit
    return jsonify({'sentiment': int(sentiment)})  # Convertir le sentiment en entier

if __name__ == '__main__':
    app.run(debug=True)
