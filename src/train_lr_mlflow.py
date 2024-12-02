import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


# Configurer l'URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Charger les données prétraitées et vectorisées
X_train = pickle.load(open('../data/processed/X_train_tfidf.pickle', 'rb'))
X_test = pickle.load(open('../data/processed/X_test_tfidf.pickle', 'rb'))
y_train = pickle.load(open('../data/processed/y_train.pickle', 'rb'))
y_test = pickle.load(open('../data/processed/y_test.pickle', 'rb'))

# Démarrer une exécution MLflow
with mlflow.start_run():
    # Créer et entraîner le modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer l'accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Enregistrer les paramètres et les métriques
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)

    # Enregistrer le modèle
    mlflow.sklearn.log_model(model, "model")
