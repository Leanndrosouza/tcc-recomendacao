import pandas as pd
import numpy as np
import logging
import re
from geopy.distance import geodesic

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

AVAILABLE_DISTRICTS = ['Jabotiana', 'Atalaia', 'Salgado Filho', 'Lamarão', 'Grageru',
                       'Luzia', 'Jardins', 'Farolândia', 'Coroa do Meio', 'Ponto Novo',
                       'Suíssa', 'Centro', 'São Conrado', 'Inácio Barbosa', 'São José',
                       'Zona de Expansão (Mosqueiro)', 'Treze de Julho',
                       'Dezoito do Forte', 'Aeroporto', 'Zona de Expansão (Aruana)',
                       'Siqueira Campos', 'Pereira Lobo', 'Jardim Centenário',
                       'Santa Maria', 'Dom Luciano', 'Olaria', 'Santos Dumont', 'América',
                       'Santo Antônio', 'Getúlio Vargas', 'Cirurgia', 'Industrial',
                       "Porto D'Antas", '18 do Forte', 'José Conrado de Araújo',
                       'Cidade Nova', 'Zona de Expansão (Robalo)', '13 De Julho',
                       'Palestina', 'Soledade', 'Japãozinho']


DISTRICT_LOCATIONS_DATAFRAME = pd.read_json(
    'app/assets/districts_location.json')

VALID_POINTS = ['point_of_interest',
 'establishment',
 'store',
 'political',
 'food',
 'health',
 'clothing_store',
 'sublocality_level_1',
 'sublocality',
 'locality',
 'finance',
 'restaurant',
 'home_goods_store',
 'general_contractor',
 'lodging',
 'place_of_worship',
 'church',
 'electronics_store',
 'school',
 'beauty_salon']

def conta_estabelecimentos(pontos_proximos):
    pontos_validos = {}
    for ponto in pontos_proximos:
        if 'types' in ponto:
            for tp in ponto['types']:
                if tp in VALID_POINTS:
                    pontos_validos[tp] = True
    return len(pontos_validos)


def load_properties():
    df = pd.read_json("app/assets/olx_location_cep_nearby.json",
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

    numbers_columns = ['value', 'rooms', 'area', 'bathrooms', 'garages', 'nearby_locations']

    for column in numbers_columns:
        df[column] = df[column].apply(str).apply(lambda x: re.sub('[^0-9]', '', x))
        df[column] = df[column].apply(lambda x: 0 if x == '' else x).astype(float)
        df[column] = df[column].astype(float)
    
    df['price_suggested'] = df['price_suggested'].astype(float)

    df['value'] = df['value'].apply(lambda x: x*1000 if x < 1000 else x)
    df['value'] = df['value'].apply(lambda x: x/1000 if x > 50000000 else x)

    df = df.replace(to_replace=0.0, value=df.median(), method='ffill')

    return df


PROPERTIES_DATAFRAME = load_properties()


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


# 'valor': [30.0],
# 'quartos': [10.0],
# 'area': [10.0],
# 'banheiros': [10.0],
# 'garagens': [10.0],
# 'bairro': [10.0],
# 'distance': [10.0]
def valid_params(request):
    # OPTIONAL PARAMS
    value = request.args.get('valor')
    rooms = request.args.get('quartos')
    area = request.args.get('area')
    bathrooms = request.args.get('banheiros')
    garages = request.args.get('garagens')
    district = request.args.get('bairro')
    latitude = request.args.get('latitude')
    longitude = request.args.get('longitude')

    if value:
        if not is_int(value):
            return 'value must be a integer'

    if rooms:
        if not is_int(rooms):
            return 'rooms must be a integer'

    if area:
        if not is_int(area):
            return 'area must be a integer'

    if bathrooms:
        if not is_int(bathrooms):
            return 'bathrooms must be a integer'

    if garages:
        if not is_int(garages):
            return 'garages must be a integer'

    if district:
        if not district in AVAILABLE_DISTRICTS:
            return 'district invalid'

    if latitude and longitude:
        if not is_float(latitude):
            return 'latitude invalid'
        if not is_float(longitude):
            return 'longitude invalid'

    return None


def is_int(x):
    try:
        int(x)
        return True
    except:
        return False

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


def prepare_params(request):
    params = {}

    value = request.args.get('valor')
    rooms = request.args.get('quartos')
    area = request.args.get('area')
    bathrooms = request.args.get('banheiros')
    garages = request.args.get('garagens')
    district = request.args.get('bairro')

    if value:
        value = float(value)
        params["value"] = value

    if rooms:
        rooms = float(rooms)
        params["garages"] = rooms

    if area:
        area = float(area)
        params["area"] = area

    if bathrooms:
        bathrooms = float(bathrooms)
        params["bathrooms"] = bathrooms

    if garages:
        garages = float(garages)
        params["garages"] = garages

    if district:
        params["district"] = district

    if len(params) < 2:
        return None, "you must provide at least two features"

    return params, None


def handle_dataframe_values(df, params):
    if not 'district' in params.keys():
        return df, params

    locations = DISTRICT_LOCATIONS_DATAFRAME

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
