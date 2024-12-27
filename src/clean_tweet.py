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

