from flask import Flask, request, jsonify
import pickle
import sys

# Ajouter le chemin du dossier src pour importer les modules
sys.path.append('src')
import preprocessing

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
    tweet_cleaned = preprocessing.clean_tweet(tweet)

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