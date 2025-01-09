# DONE

- Sauvegarder le vectorizer / prétraitement pour utiliser dans l'API

- Ajouter paramètre nom sur reduce_dim

- Faire fonctionner l'API Flask

- AJouter un Embedding dans le LSTM avec Word2vec
model.add_layer(Embedding...)

- Utiliser MFlow dans tous les modèles
Empaqueter tous dans MFlow (with MLFlow)

- LSTM avec Fasttext

- Optimisation du LSTM

- TOC + Structure (jupyterlab)

- Récupérer les metrics des données

- Tableau comparatif 

- Faire fonctionner BERT (1% des données).
Faire via le evaluate_model (cf fleuryc)


- Améliorer les performances des modèles LSTM


- Relancer les modèles sur 10% des données

- pipeline d’entraînement des modèles reproductible

- sérialisé et stocké les modèles créés dans un registre centralisé afin de pouvoir facilement les réutiliser.

- formalisé des mesures et résultats de chaque expérimentation, afin de les analyser et de les comparer


- Setup d'un compte pythonanywhere

- Relancer BERT sur 1% des données

- déployé le modèle de machine learning sous forme d'API (via Flask par exemple) et cette API renvoie bien une prédiction correspondant à une demande. 



# TODO


- pipeline de déploiement continu, afin de déployer l'API sur un serveur d'une plateforme Cloud. 

- tests unitaires automatisés (par exemple avec pyTest)

- réalisé l'API indépendamment de l'application qui utilise le résultat de la prédiction. 

- stratégie de suivi de la performance du modèle. Choix d’utiliser Azure Application Insight pour le suivi de traces de prédictions non conformes et de déclenchement d’alertes

- stockage d’événements relatifs aux prédictions réalisées par l’API et une gestion d’alerte en cas de dégradation significative de la performance. Mise en oeuvre sur Azure Application Insight de de traces relatives à des prédictions non conformes, paramétrage de déclenchement d’alertes et exécution des alertes envoyées par mail ou SMS

- Stabilité du modèle dans le temps et défini des actions d’amélioration de sa performance. Présentation dans le blog d’une démarche qui pourrait être mise en oeuvre pour l’analyse de ces statistiques et l’amélioration du modèle dans le temps




# Bilan 




- Taille de l'environnement virtuel (Plusieurs GiO !!). 
J'ai dû installé un environnement simplifié à ce stade.

- PythonAnywhere ==> prb de taille 

- Solution : AWS





