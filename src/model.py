import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Lambda
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification


def create_mlp_model(input_shape):
    """
    Crée un modèle MLP (Multi-Layer Perceptron) avec les couches suivantes :
    - Dense(64, activation='relu')
    - Dropout(0.4)
    - Dense(32, activation='relu')
    - Dropout(0.3)
    - Dense(1, activation='sigmoid')

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle MLP compilé.
    """

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_cnn_model(input_shape):
    """
    Crée un modèle CNN (Convolutional Neural Network) avec les couches suivantes :
    - Conv1D(filters=32, kernel_size=3, activation='relu')
    - MaxPooling1D(pool_size=2)
    - Conv1D(filters=64, kernel_size=3, activation='relu')
    - MaxPooling1D(pool_size=2)
    - Flatten()
    - Dense(units=1, activation='sigmoid')

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle CNN compilé.
    """

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def create_lstm_model(input_shape):
    """
    Crée un modèle LSTM (Long Short-Term Memory) avec les couches suivantes :
    - Input(shape=input_shape)
    - Lambda(lambda x: tf.expand_dims(x, axis=1))  # Ajout d'une dimension temporelle
    - LSTM(units=64)
    - Dropout(0.2)
    - Dense(units=1, activation='sigmoid')

    Args:
        input_shape : La forme des données d'entrée du modèle.

    Returns:
        Le modèle LSTM compilé.
    """

    model = Sequential()
    model.add(Input(shape=input_shape))  # Couche d'entrée avec la dimension des embeddings
    model.add(Lambda(lambda x: tf.expand_dims(x, axis=1)))  # Ajout d'une dimension temporelle
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
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