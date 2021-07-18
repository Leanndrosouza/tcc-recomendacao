import pandas as pd
import numpy as np
import logging
import re
from geopy.distance import geodesic

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


def valid_params(args):
    value = args.get('value')
    rooms = args.get('rooms')
    area = args.get('area')

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


def prepare_params(args):
    value = float(args.get('value'))  # int
    rooms = float(args.get('rooms'))  # int
    area = float(args.get('area'))  # int

    params = {
        "value": value,
        "rooms": rooms,
        "area": area
    }

    # OPTIONAL

    bathrooms = args.get('bathrooms')
    garages = args.get('garages')
    district = args.get('district')

    if bathrooms:
        bathrooms = float(bathrooms)
        params["bathrooms"] = bathrooms

    if garages:
        garages = float(garages)
        params["garages"] = garages

    if district:
        params["district"] = district

    return params, None


def conta_estabelecimentos(pontos_proximos):
    pontos_validos = {}
    for ponto in pontos_proximos:
        if 'types' in ponto:
            for tp in ponto['types']:
                pontos_validos[tp] = True
    return len(pontos_validos)


def load_properties():
    df = pd.read_json("assets/olx_location_cep_nearby.json",
                      orient="records", convert_dates=False)
    df = df.drop(["link", "descricao",
                 "created_at", "codigo"], axis=1)
    df = pd.concat([
        df["_id"].apply(pd.Series),
        df.drop(["_id", "data_publicacao", "caracteristicas"], axis=1),
        df['caracteristicas'].apply(pd.Series)
    ], axis=1)
    df = df.drop(["tipo", "cep", "municipio", "logradouro", "detalhes_do_imovel",
                 "detalhes_do_condominio", "condominio", "iptu"], axis=1)
    df = df.rename({
        '$oid': 'id',
        'preco': 'value',
        'quartos': 'rooms',
        'area_util': 'area',
        'banheiros': 'bathrooms',
        'vagas_na_garagem': 'garages',
        'localizacao': 'location',
        'bairro': 'district_name',
        'previsao': 'price_suggested'
    }, axis=1)

    df["nearby_locations"] = df["nearby_locations"].apply(conta_estabelecimentos)

    numbers_columns = ['value', 'rooms', 'area', 'bathrooms', 'garages']

    for column in numbers_columns:
        df[column] = df[column].apply(str).apply(lambda x: re.sub('[^0-9]', '', x))
        df[column] = df[column].apply(lambda x: 0 if x == '' else x).astype(float)
        df[column] = df[column].astype(float)
    
    df['price_suggested'] = df['price_suggested'].astype(float)

    df['value'] = df['value'].apply(lambda x: x*1000 if x < 1000 else x)
    df['value'] = df['value'].apply(lambda x: x/1000 if x > 50000000 else x)

    df = df.replace(to_replace=0.0, value=df.median(), method='ffill')

    return df

def handle_dataframe_values(df, params):
    if not 'district' in params.keys():
        return df, params

    locations = pd.read_json('assets/districts_location.json')

    location = locations[params['district']]

    df['distance'] = df.apply(lambda row: calculate_distance(
        location['lat'],
        location['lng'],
        row['location']['lat'],
        row['location']['lng']
    ), axis=1)

    df['district'] = df['district_name'].apply(
        lambda x: float(x == params['district'])
    )

    params['district'] = 1.0
    params['distance'] = 0.0

    return df, params

def calculate_distance(lat1, lng1, lat2, lng2):
    return geodesic((lat1, lng1), (lat2, lng2)).km

def convert_to_output(dict):
    dict = rename_key(dict, "banheiros", "bathrooms")
    dict = rename_key(dict, "bairro", "district_name")
    dict = rename_key(dict, "preco_estimado", "price_suggested")
    dict = rename_key(dict, "similaridade", "similarity")
    dict = rename_key(dict, "valor", "value")
    dict = rename_key(dict, "quartos", "rooms")
    dict = rename_key(dict, "garagens", "garages")
    dict = rename_key(dict, "locais_proximos", "nearby_locations")
    if 'distance' in dict.keys():
        dict = rename_key(dict, "distancia", "distance")
    
    if 'district' in dict.keys():
        dict.pop("district")

    dict.pop("categoria")
    dict.pop("location")
    dict.pop("similarity_index")

    return dict

def rename_key(dict, n, o):
    dict[n] = dict.pop(o)
    return dict