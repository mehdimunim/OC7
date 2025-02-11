# Détection de Bad Buzz avec le Deep Learning

## Description

L'entreprise "Air Paradis" nous commissione pour développer un modèle de Deep Learning capable de détecter les bad buzz (opinions négatives) sur les réseaux sociaux. Le modèle sera entraîné sur des données opensource (tweets)[https://www.kaggle.com/kazanova/sentiment140]. L'objectif est de construire un modèle capable de prédire avec précision si un tweet exprime un sentiment positif ou négatif.

## Étapes

-  Collecter un jeu de données de tweets étiquetés (positif/négatif) et le préparer pour l'entraînement du modèle (nettoyage, prétraitement, vectorisation).
- Concevoir et implémenter un modèle de Deep Learning pour la classification de sentiment.
- Entraîner le modèle sur les données préparées et évaluer ses performances en utilisant des métriques appropriées.
- Optimiser les hyperparamètres du modèle pour améliorer ses performances.
- Déployer le modèle sur une plateforme cloud et le rendre accessible via une API.
- Mettre en place un système de surveillance pour suivre les performances du modèle en production.

## Structure du projet

```
├── notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelisation.ipynb
│
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

- **`notebooks/`**: Contient les notebooks Jupyter pour chaque étape du projet.
- **`data/`**: Contient les données du projet (brutes, prétraitées, externes).
- **`models/`**: Contient les modèles entraînés.
- **`src/`**: Contient les scripts Python pour le prétraitement, la modélisation et l'évaluation.
- **`requirements.txt`**: Liste des packages Python nécessaires.
- **`README.md`**
- **`.gitignore`**: Fichier qui spécifie les fichiers et dossiers à ignorer par Git.

## Installation

1. Cloner le dépôt GitHub
2. Créer un environnement virtuel Python:
   ```bash
   python3 -m venv .venv
   ```
3. Activer l'environnement virtuel:
   ```bash
   source .venv/bin/activate  # Sur Linux/macOS
   .venv\Scripts\activate  # Sur Windows
   ```
4. Installer les packages nécessaires avec `pip install -r requirements.txt`

## Exécution

1.  Ouvrir les notebooks Jupyter dans l'ordre.
2.  Exécuter les cellules des notebooks pour effectuer les différentes étapes du projet.
3.  Pour utiliser l'API streamlit (prédiction de tweet ) : 

## Contributeurs

- Mehdi MUNIM

## Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.
