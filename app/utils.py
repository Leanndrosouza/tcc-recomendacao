import pandas as pd
import numpy as np
import logging
import re

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def mahalanobis_algorithm(df, request_dict, weights_dict):
    # handle data
    request_keys = list(request_dict.keys())
    request_values = list(request_dict.values())

    # tranform weights dict in Dataframe
    weights = pd.DataFrame(data=weights_dict)

    # prepare data
    data_arr = df[request_keys].values
    data_mean = np.mean(data_arr, axis=0)
    data_std = np.std(data_arr, axis=0, ddof=1)
    data_arr -= data_mean
    data_arr /= data_std

    query_arr = np.array([request_values])
    query_arr -= data_mean
    query_arr /= data_std

    weight = np.diag(weights[request_keys].values[0])

    # applying mahalanobis algorithm
    distances = (pd.DataFrame(abs(np.diag(
        (data_arr - query_arr) @ weight @ np.linalg.inv(np.cov(data_arr.transpose())) @ (
            ((data_arr - query_arr) @ weight).transpose()))) ** .5)).sort_values(by=0)
    newdistances = pd.DataFrame(distances)
    sorted_values = newdistances.sort_values(by=0)
    return sorted_values


def valid_params(request):
    value = request.args.get('value')
    rooms = request.args.get('rooms')
    area = request.args.get('area')

    if not value:
        return 'value is a required field'

    if not is_int(value):
        return 'value must be a integer'

    if not rooms:
        return 'rooms is a required field'

    if not is_int(rooms):
        return 'rooms must be a integer'

    if not area:
        return 'area is a required field'

    if not is_int(area):
        return 'area must be a integer'

    return None


def is_int(x):
    try:
        int(x)
        return True
    except:
        return False


def prepare_params(request):
    value = float(request.args.get('value'))  # int
    rooms = float(request.args.get('rooms'))  # int
    area = float(request.args.get('area'))  # int

    return {"value": value, "rooms": rooms, "area": area}, None


def load_properties():
    df = pd.read_json("app/assets/olx.json", orient="records", convert_dates=False)
    df = df.drop(["titulo", "link", "descricao", "created_at", "codigo"], axis=1)
    df = pd.concat([df.drop(["_id", "data_publicacao", "caracteristicas"], axis=1), df['caracteristicas'].apply(pd.Series)], axis=1)
    df = df.drop(["tipo", "cep", "municipio", "logradouro", "detalhes_do_imovel", "detalhes_do_condominio", "condominio", "iptu"], axis=1)
    df = df.dropna()
    df = df.rename({'preco': 'value', 'quartos': 'rooms', 'area_util': 'area'}, axis=1)

    df['value'] = df['value'].apply(lambda x: re.sub('[^0-9]','', x))
    df['value'] = df['value'].astype(float)
    df['rooms'] = df['rooms'].apply(lambda x: re.sub('[^0-9]','', x))
    df['rooms'] = df['rooms'].astype(float)
    df['area'] = df['area'].apply(lambda x: re.sub('[^0-9]','', x))
    df['area'] = df['area'].astype(float)

    return df
