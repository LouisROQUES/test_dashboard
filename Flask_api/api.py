# coding=utf-8

from io import BytesIO
import joblib
import requests
from flask import jsonify
import pandas as pd
import json

from flask import Flask
app = Flask(__name__)

@app.route('/test')
def test():
    return jsonify({'test': 'ok'})

@app.route('/importances')
def importances():
    """
    Fonction permettant de charger le modèle de prédiction
    :return: modèle
    """
    mLink = 'https://github.com/LouisROQUES/test_dashboard/blob/master/gbc_model.pkl?raw=true'
    mfile = BytesIO(requests.get(mLink).content)
    xgbc_model = joblib.load(mfile)
    importances = xgbc_model.feature_importances_
    return jsonify(dict(enumerate(importances.flatten(), 1)))

@app.route('/predict_prob/<int:number_input1>')
def predict_prob(number_input1):
    """
    Cette fonction sert à prédire ...
    :param X_test: le vecteur de données pour un individu (liste)
    :return: prédiction de type float
    """
    mLink = 'https://github.com/LouisROQUES/test_dashboard/blob/master/gbc_model.pkl?raw=true'
    mfile = BytesIO(requests.get(mLink).content)
    xgbc_model = joblib.load(mfile)
    url = 'https://drive.google.com/file/d/1060KLYzDLZe77dCyAjYUVOWsrSQfLHHa/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    data = pd.read_csv(path)
    data_process = data[data['SK_ID_CURR']==number_input1]
    data_process = data_process.drop(['SK_ID_CURR'], axis=1)
    pred = xgbc_model.predict_proba(data_process)[:,1]
    return jsonify(dict(enumerate(pred.flatten(), 1)))

if __name__ == '__main__':
   app.run(debug = True)

