from io import BytesIO
import joblib
import requests


def load_model():
    """
    Fonction permettant de charger le modèle de prédiction
    :return: modèle
    """
    mLink = 'https://github.com/LouisROQUES/test_dashboard/blob/master/Model/gbc_model.pkl?raw=true'
    mfile = BytesIO(requests.get(mLink).content)
    xgbc_model = joblib.load(mfile)
    return xgbc_model

def predict_prob(X_test):
    """
    Cette fonction sert à prédire ...
    :param X_test: le vecteur de données pour un individu (liste)
    :return: prédiction de type float
    """
    mLink = 'https://github.com/LouisROQUES/test_dashboard/blob/master/Model/gbc_model.pkl?raw=true'
    mfile = BytesIO(requests.get(mLink).content)
    xgbc_model = joblib.load(mfile)
    pred = xgbc_model.predict_proba(X_test)[:,1]
    return pred

