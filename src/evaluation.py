import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score, roc_curve, auc)
from tensorflow.keras.models import Sequential
import tensorflow as tf

def evaluer_modele_bert(model, X_test, y_test):
    """
    Évalue un modèle BERT en affichant les métriques d'évaluation et la courbe ROC.

    Args:
        model : Le modèle BERT à évaluer.
        X_test : Les données de test tokenisées.
        y_test : Les vraies étiquettes des données de test.
    """

    # Début du chronométrage
    start_time = time.time()

    # Prédire les probabilités et les labels
    y_pred = model.predict({"input_ids": X_test["input_ids"], "attention_mask": X_test["attention_mask"]})
    y_pred_proba = [float(x[1]) for x in tf.nn.softmax(y_pred.logits)]
    y_pred_label = [0 if x[0] > x[1] else 1 for x in tf.nn.softmax(y_pred.logits)]

    # Fin du chronométrage
    end_time = time.time()

    # Calculer le temps de prédiction
    predict_time = end_time - start_time

    # Calculer les métriques d'évaluation
    accuracy = accuracy_score(y_test, y_pred_label)
    precision = precision_score(y_test, y_pred_label)
    recall = recall_score(y_test, y_pred_label)
    f1 = f1_score(y_test, y_pred_label)
    cm = confusion_matrix(y_test, y_pred_label)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Afficher les métriques
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_precision:.4f}")

    # Afficher la matrice de confusion
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()

    # Calculer et afficher la courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()

    # Retourner les métriques et le temps de prédiction
    return accuracy, precision, recall, f1, predict_time

def evaluer_modele(model, X_test, y_test):
    """
    Évalue un modèle de classification en affichant les métriques
    d'évaluation (accuracy, precision, recall, F1-score) et la courbe ROC.

    Args:
        model : Le modèle à évaluer.
        X_test : Les données de test.
        y_test : Les vraies étiquettes des données de test.
    """
    start_time = time.time()
    # Obtenir les prédictions du modèle
    if isinstance(model, Sequential):  # Vérifier si le modèle est un modèle Keras
        y_pred = model.predict(X_test).flatten()
        y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en classes prédites si nécessaire
    else:
        y_pred = model.predict(X_test)
    end_time = time.time()
    # Calculer le temps de prédiction
    prediction_time = end_time - start_time

    # Calculer les métriques d'évaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    # Afficher les métriques
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")

    # Calculer la courbe ROC et l'AUC (si possible)
    try:
        if isinstance(model, Sequential):
            y_pred_proba = model.predict(X_test).flatten()  # Obtenir les probabilités pour les modèles Keras
        else:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Afficher la matrice de confusion et la courbe ROC côte à côte
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Matrice de confusion
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Matrice de confusion')
        axes[0].set_ylabel('Vraie classe')
        axes[0].set_xlabel('Classe prédite')

        # Courbe ROC
        axes[1].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        axes[1].plot([0, 1], [0, 1], 'k--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Taux de faux positifs')
        axes[1].set_ylabel('Taux de vrais positifs')
        axes[1].set_title('Courbe ROC')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.show()

    except AttributeError:
        print("Le modèle ne supporte pas la méthode predict_proba. Impossible d'afficher la courbe ROC.")

    # Retourner les métriques et le temps de prédiction
    return accuracy, precision, recall, f1, prediction_time

