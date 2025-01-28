import pytest
from app import app  # Importer ton application Flask

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_positive_sentiment(client):
    response = client.post('/predict', data={'tweet': 'This is a great movie!'})
    assert response.status_code == 200
    assert response.json['sentiment'] == 'positif'

def test_predict_negative_sentiment(client):
    response = client.post('/predict', data={'tweet': 'This movie is terrible.'})
    assert response.status_code == 200
    assert response.json['sentiment'] == 'négatif'

def test_predict_empty_tweet(client):
    response = client.post('/predict', data={'tweet': ''})
    assert response.status_code == 400  # Ou le code d'erreur que tu as défini
