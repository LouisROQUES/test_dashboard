import pandas as pd
import numpy as np


def load_data():
    """
    Fonction permettant de charger le jeu données contenant les informations clients
    :return: dataframe contenant les infos clients
    """
    url = 'https://drive.google.com/file/d/1lBO_5ektxTS9Dj6xGhqEY64EFTglXz12/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    raw_data = pd.read_csv(path)
    raw_data['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    raw_data['CREDIT_INCOME_PERCENT'] = raw_data['AMT_CREDIT'] / raw_data['AMT_INCOME_TOTAL']
    raw_data['ANNUITY_INCOME_PERCENT'] = raw_data['AMT_ANNUITY'] / raw_data['AMT_INCOME_TOTAL']
    raw_data['CREDIT_TERM'] = raw_data['AMT_ANNUITY'] / raw_data['AMT_CREDIT']
    raw_data['DAYS_EMPLOYED_PERCENT'] = raw_data['DAYS_EMPLOYED'] / raw_data['DAYS_BIRTH']
    raw_data['DAYS_BIRTH'] = raw_data['DAYS_BIRTH'] *-1
    return raw_data

def load_data_preprocess():
    """
    Fonction permettant de charger le jeu données contenant les informations clients preprocessées
    :return: dataframe contenant les infos clients préprocessées
    """
    url = 'https://drive.google.com/file/d/1060KLYzDLZe77dCyAjYUVOWsrSQfLHHa/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    data = pd.read_csv(path)
    return data

def get_data(number_input1, data_input):
    """
    Fonction permettant de récupérer les données du client brutes
    :param number_input1: n° client rentré par le chargé de clientèle
    :param data_input: base de données sur les infos clients
    :return: dataframe contenant uniquement les informations du client
    """
    data = data_input[data_input['SK_ID_CURR']==number_input1]
    data = data.drop(['SK_ID_CURR'], axis=1)
    return data

def get_raw_data(number_input1, raw_data_input):
    """
    Fonction permettant de récupérer les données du client préprocessées
    :param number_input1: n° client rentré par le chargé de clientèle
    :param data_input: base de données sur les infos clients
    :return: dataframe contenant uniquement les informations du client préprocessées
    """
    data = raw_data_input[raw_data_input['SK_ID_CURR']==number_input1]
    data = data.drop(['SK_ID_CURR'], axis=1)
    return data

def load_data_train():
    """
    Fonction permettant de charger le jeu données contenant les informations clients d'entrainement
    :return: dataframe contenant les infos clients
    """
    url1 = 'https://drive.google.com/file/d/1Liiw2mZ7Sr2hiw0hm75DGMTMRfMazKj9/view?usp=sharing'
    path1 = 'https://drive.google.com/uc?export=download&id=' + url1.split('/')[-2]
    raw_data1 = pd.read_csv(path1)
    url2 = 'https://drive.google.com/file/d/1arBQJBJ5I-t6ShB8CA6ZTzKtxDuzP58X/view?usp=sharing'
    path2 = 'https://drive.google.com/uc?export=download&id=' + url2.split('/')[-2]
    raw_data2 = pd.read_csv(path2)
    app_train = pd.concat([raw_data1, raw_data2])
    app_train["TARGET"] = app_train["TARGET"].replace([0, 1], ['paiment_ok', 'deafaut_paiement'])
    return app_train

def load_train_set():
    """
    Fonction permettant de charger le jeu de données d'entrainement du modèle pour visualisation du feature importance
    :return: dataframe contenant le jeu de données d'entrainement du modèle
    """
    url = 'https://drive.google.com/file/d/1RpyhrDoXDr_La6V3fLTl3ymikkQsXSjD/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
    train_set = pd.read_csv(path)
    return train_set