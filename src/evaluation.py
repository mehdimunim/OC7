from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.models import Sequential

def evaluer_modele(model, X_test, y_test):
    """
    Évalue un modèle de classification en affichant les métriques
    d'évaluation (accuracy, precision, recall, F1-score) et la courbe ROC.

    Args:
        model : Le modèle à évaluer.
        X_test : Les données de test.
        y_test : Les vraies étiquettes des données de test.
    """

    # Obtenir les prédictions du modèle
    if isinstance(model, Sequential):  # Vérifier si le modèle est un modèle Keras
        y_pred = model.predict(X_test).flatten()
        y_pred = (y_pred > 0.5).astype(int)  # Convertir les probabilités en classes prédites si nécessaire
    else:
        y_pred = model.predict(X_test)

    # Calculer les métriques d'évaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Afficher les métriques
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

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
        axes[1].set_ylabel('Taux