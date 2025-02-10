import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Lambda
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification

def create_mlp_model(input_shape):
    """
    Crée un modèle MLP (Multi-Layer Perceptron) plus complexe.

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle MLP compilé.
    """

    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_shape=input_shape))  # Augmentation du nombre de neurones
    model.add(Dropout(0.2))  # Diminution du dropout
    model.add(Dense(units=128, activation='relu'))  # Ajout d'une couche cachée
    model.add(Dropout(0.2))  # Diminution du dropout
    model.add(Dense(units=64, activation='relu')) 
    model.add(Dropout(0.1))  # Diminution du dropout
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle avec un optimiseur plus performant
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_cnn_model(input_shape):
    """
    Crée un modèle CNN (Convolutional Neural Network) plus complexe.

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle CNN compilé.
    """

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))  # Augmentation du nombre de filtres
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))  # Augmentation du nombre de filtres
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle avec un optimiseur plus performant
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_lstm_model(input_shape, embedding_matrix):
    """
    Crée un modèle LSTM (Long Short-Term Memory) plus complexe.

    Args:
      input_shape : La forme des données d'entrée du modèle (input_dim, output_dim).
      embedding_matrix: La matrice

    Returns:
      Le modèle LSTM compilé.
    """

    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], 
                        output_dim=input_shape[1], 
                        weights=[embedding_matrix],
                        trainable=False))  # Les embeddings sont figés
    model.add(LSTM(units=128, return_sequences=True))  # Augmentation du nombre d'unités LSTM et ajout de return_sequences
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))  # Ajout d'une couche LSTM supplémentaire
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle avec un optimiseur plus performant
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
    
def create_distilbert_model(model_name="distilbert-base-uncased"):
    """
    Crée et compile un modèle DistilBERT pour la classification de séquences,
    avec un tokenizer associé.

    Args:
        model_name (str, optional): Le nom du modèle pré-entraîné DistilBERT à charger
            depuis Hugging Face Transformers.  Par défaut, "distilbert-base-uncased".
            Vous pouvez utiliser d'autres modèles comme "distilbert-base-cased",
            "distilbert-base-multilingual-cased", etc.

    Returns:
        tuple: Un tuple contenant:
            - model (TFDistilBertForSequenceClassification): Le modèle DistilBERT compilé.
            - tokenizer (DistilBertTokenizer): Le tokenizer DistilBERT pré-entraîné.
    """

    # Charger le tokenizer pré-entraîné DistilBERT.
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Charger le modèle DistilBERT pré-entraîné pour la classification de séquences.
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Définir la fonction de perte.  SparseCategoricalCrossentropy est utilisée lorsque
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Compiler le modèle.  La compilation configure le processus d'entraînement.
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    return model, tokenizer
    
def create_bert_model(model_name="bert-base-uncased"):
    """
    Crée un modèle BERT.

    Args:
        model_name : Le nom du modèle BERT à utiliser.

    Returns:
        Le modèle BERT compilé.
    """

    # Charger le tokenizer et le modèle pré-entraîné
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Compiler le modèle
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  


    return model, tokenizer


def create_use_model(input_shape):
    """
    Crée un modèle USE (Universal Sentence Encoder).

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle USE compilé.
    """

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model  

def save_model_weights(model, model_name, path="../models/"):
    """
    Sauvegarde les poids d'un modèle.

    Args:
        model : Le modèle dont les poids doivent être sauvegardés.
        model_name : Le nom du fichier pour sauvegarder les poids.
        path : Le chemin du dossier où sauvegarder les poids.
    """
    # Créer le dossier s'il n'existe pas
    os.makedirs(path, exist_ok=True)

    # Sauvegarder les poids du modèle
    model.save_weights(os.path.join(path, f"{model_name}.h5"))