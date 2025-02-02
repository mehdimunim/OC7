import streamlit as st
import requests

# Fonction pour prédire le sentiment
def predict_sentiment(tweet):
    url = 'http://3.83.30.83:8000/predict'
    data = {'tweet': tweet}
    response = requests.post(url, data=data)
    return response.json()['sentiment']

# Interface Streamlit
st.title("Analyse de sentiment")

tweet = st.text_input("Entrez un tweet")

if st.button("Prédire"):
    if tweet:
        sentiment = predict_sentiment(tweet)
        st.write(f"Sentiment prédit: {sentiment}")
    else:
        st.warning("Veuillez entrer un tweet.")
