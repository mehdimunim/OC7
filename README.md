# Détection de Bad Buzz avec le Deep Learning

## Description

Ce projet vise à développer un modèle de Deep Learning capable de détecter les bad buzz (opinions négatives) sur les réseaux sociaux, en se concentrant sur l'analyse de sentiment des tweets. L'objectif est de construire un modèle capable de prédire avec précision si un tweet exprime un sentiment positif ou négatif.

## Objectifs

*   **Collecte et préparation des données**: Collecter un jeu de données de tweets étiquetés (positif/négatif) et le préparer pour l'entraînement du modèle (nettoyage, prétraitement, vectorisation).
*   **Développement d'un modèle de Deep Learning**: Concevoir et implémenter un modèle de Deep Learning pour la classification de sentiment.
*   **Entraînement et évaluation du modèle**: Entraîner le modèle sur les données préparées et évaluer ses performances en utilisant des métriques appropriées.
*   **Optimisation du modèle**: Optimiser les hyperparamètres du modèle pour améliorer ses performances.
*   **Déploiement du modèle**: Déployer le modèle sur une plateforme cloud et le rendre accessible via une API.
*   **Surveillance du modèle**: Mettre en place un système de surveillance pour suivre les performances du modèle en production.

## Structure du projet

```
├── notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modelisation.ipynb
│   └── 04_evaluation.ipynb
├── data
│   ├── raw
│   │   └── dataset.csv
│   ├── processed
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external
│       └── donnees_externes.csv
├── models
│   └── meilleur_modele.pkl
├── src
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
```

*   **`notebooks/`**: Contient les notebooks Jupyter pour chaque étape du projet.
*   **`data/`**: Contient les données du projet (brutes, prétraitées, externes).
*   **`models/`**: Contient les modèles entraînés.
*   **`src/`**: Contient les scripts Python pour le prétraitement, la modélisation et l'évaluation.
*   **`requirements.txt`**: Liste des packages Python nécessaires.
*   **`README.md`**: Ce fichier.
*   **`.gitignore`**: Fichier qui spécifie les fichiers et dossiers à ignorer par Git.

## Installation

1.  Cloner le dépôt GitHub.
2.  Créer un environnement virtuel Python.
3.  Installer les packages nécessaires avec `pip install -r requirements.txt`.

## Exécution

1.  Ouvrir les notebooks Jupyter dans l'ordre.
2.  Exécuter les cellules des notebooks pour effectuer les différentes étapes du projet.

## Remarques

*   Ce projet utilise Python et des librairies comme pandas, scikit-learn, TensorFlow, Transformers et MLflow.
*   Le jeu de données utilisé est un jeu de données de tweets étiquetés (positif/négatif).
*   Le modèle de Deep Learning utilisé est un modèle BERT fine-tuné pour la classification de sentiment.

## Contributeurs

*   Mehdi MUNIM

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

