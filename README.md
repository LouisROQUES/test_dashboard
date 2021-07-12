# Dashboard pour la prédiction de capacité de capacité de remboursement d'un prêt

Ce projet consiste à créer à partir d'une base de données, un dashboard interactif permettant d'interpréter les prédictions faites par un modèle entrainé de probabilité de remboursement d'un prêt. 

# Sommaire

- Data
- Modèle
- App

## Data

Ici les données sont issues d'une compétition Kagggle "Home Credit Default Risk". Les données sources principalement utilisées sont application_train.csv et application test.csv. Au cours de l'avancement du projet, d'autres dataset dérivés de ceux-ci sont loader. Des datasets nettoyés ou encore préprocessés.
Les principaux traitements des dataset sont :
- Nettoyage des features avec un taux de valeurs présentes <80%
- Remplacement de certaines valeurs aberrantes notamment dans la feature DAYS_EMPLOYED
- Features engineering 
- OneHotEncoder des features catégorielles
- Imputation des données manquantes par la médiane pour les features numériques et par le most_frequent pour les variables catégorielles
- MinMaxScaler des features numériques
- Equilibrage des classes car une des deux target est beaucoup plus représentée, donc risque d'overfitting sur une des classes

## Le modèle

Le modèle utilisé est un modèle GradientBoostingClassifier, une recherche du meilleur modèle a été effectué dans un notebook séparé. Ce modèle a été comparé à une régression logistique, un RandomForestClassifier, et un LGMB Classifier. L'évaluation du meilleur modèle s'est basé principalement sur les scores ROC-AUC. On regardera également la matrice de confusion du modèle qui donnera une information notammet sur les bonnes préictions des deux classes, sachant que l'intérêt métier est de déceller le plus de clients ne pouvant rembourser de prêt.

## App

Une app sous forme de dashboard interactif a été développé avec Streamlit. Ce dashboard permet à partir d'un numéro client de recherche les informations de celui-ci ainsi que de prédire une probabilité de remboursement d'un prêt. Ensuite des graphiques sont générés pour expliquer le score obtenu et, une visualisation des informations client importantes comparés aux "bons payeurs" et aux "mauvais payeurs". 

Pour cette applications, les images et le modèle sont chargées de streamlit, et les datasets sont chargés de Google Drive, GitHub ne permettant pas d'héberger des fichiers aussi volumineux. 

Les différentes fonctions de cette applications sont présentes dans des modules séparés:
- Loading_data ; chargements des différentes dataset
- Model : chargement de modèle, et prédiction du score
- Visualisation : génération de graphiques ou de tableaux de données
- Main : reprend les différentes fonctions en intégrant du streamlit pour obtenir une interface web simplifiée



