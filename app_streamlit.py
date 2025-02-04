import streamlit as st
import requests
import boto3
import os

# Fonction pour prédire le sentiment
def predict_sentiment(tweet):
    url = 'http://3.83.30.83:8000/predict'
    data = {'tweet': tweet}
    response = requests.post(url, data=data)
    return response.json()['sentiment']

# Interface Streamlit
st.title("Analyse de sentiment")

if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

tweet = st.text_input("Entrez un tweet")

if st.button("Prédire"):
    if tweet:
        sentiment = predict_sentiment(tweet)
        st.write(f"Sentiment prédit: {sentiment}")
    else:
        st.warning("Veuillez entrer un tweet.")

if st.button("Signaler une erreur"):
    st.session_state.error_count += 1
    st.write(f"Nombre d'erreurs signalées: {st.session_state.error_count}")

    if st.session_state.error_count >= 1:
        # Envoyer une alerte à CloudWatch
        # Pas besoin de spécifier les identifiants AWS ici
        client = boto3.client('cloudwatch')

        response = client.put_metric_data(
            Namespace='AnalyseSentiment',
            MetricData=[
                {
                    'MetricName': 'ErreursSignalees',
                    'Value': 1,
                    'Unit': 'Count'
                },
            ]
        )

        st.warning("Alerte CloudWatch envoyée!")
        st.session_state.error_count = 0  # Réinitialiser le compteur
