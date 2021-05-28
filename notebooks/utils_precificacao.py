import pandas as pd
import numpy as np
import json
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import math

realty_attributes = ['Área de serviço', 'Armários no quarto', 'Armários na cozinha','Mobiliado', 'Ar condicionado', 'Churrasqueira', 'Varanda','Academia', 'Piscina', 'Quarto de serviço']

condominium_attributes = ['Condomínio fechado', 'Elevador','Segurança 24h','Portaria', 'Permitido animais','Academia','Piscina','Salão de festas']

description_keywords = ['varanda gourmet', 'sombra', 'lançamento', 'suíte', 'suite']

def handle_dataframe(path):
    df = pd.read_json(path, orient="records", convert_dates=False)
    df['preco'] = df['preco'].replace('[R\$,.]', '', regex=True).astype(float)
    df2 = pd.DataFrame(df.caracteristicas.apply(pd.Series))
    df = pd.concat([df, df2], axis=1)
    df['condominio'] = df['condominio'].replace('[R\$,.]', '', regex=True).astype(float)
    df['iptu'] = df['iptu'].replace('[R\$,.]', '', regex=True).astype(float)
    df['area_util'] = df['area_util'].replace('[m²]', '', regex=True).astype(float)
    df['banheiros'] = df['banheiros'].apply(lambda x: 5 if x == '5 ou mais' else x).astype(float)
    df['quartos'] = df['quartos'].apply(lambda x: 5 if x == '5 ou mais' else x).astype(float)
    df['vagas_na_garagem'] = df['vagas_na_garagem'].apply(lambda x: 5 if x == '5 ou mais' else x).astype(float)
    
    df = df.drop(['caracteristicas', 'created_at', 'localizacao'], axis=1)
    
    return df

def handle_attributes(df, columns=['preco', 'bairro' ,'area_util', 'quartos', 'banheiros', 'vagas_na_garagem', 'condominio', 'iptu']):
    df_train = df[columns]
    df_train = df_train.fillna(df_train.median())
    df_train['preco'] = df_train['preco'].replace(to_replace=0, method='ffill')
    return df_train

def test_model(model, X, y):
    results = []
    results_values = []
    kf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=0)
    for train_lines, valid_lines in kf.split(X):

        X_train, X_valid = X.iloc[train_lines], X.iloc[valid_lines]
        y_train, y_valid = y.iloc[train_lines], y.iloc[valid_lines]
        model.fit(X_train,y_train)
        p = model.predict(X_valid)
        rms = sqrt(mean_squared_error(y_valid, p))
        results.append(rms)
        results_values.append(p - y_valid)
    return results, results_values


def train_model(model, df, columns, target):
    X = df[columns]
    y = df[target]

    model.fit(X,y)
    return model

def apply_model(model, df, columns):
    X_prev = df[columns]
    df['previsao'] = np.round(model.predict(X_prev))
    if 'preco' in columns:
        df['erro absoluto'] = df['preco'] -  df['previsao']
        df['erro relativo'] = np.round((df['preco'] -  df['previsao'])/df['preco'], 2)
    return df