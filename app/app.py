from flask import Flask, request
from utils import mahalanobis_algorithm, valid_params, prepare_params, PROPERTIES_DATAFRAME, handle_dataframe_values
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def error_response(message, code):
    return {"error": message}, code

@app.route('/')
def hello():
    error = valid_params(request)

    if error != None:
        return error_response(error, 400)

    params, error = prepare_params(request)

    if error != None:
        return error_response(error, 400)

    df = PROPERTIES_DATAFRAME

    df, params = handle_dataframe_values(df, params)

    weights = {
        'value': [30.0],
        'rooms': [10.0],
        'area': [10.0],
        'bathrooms': [10.0],
        'garages': [10.0],
        'district': [10.0],
        'distance': [10.0]
    }

    sorted_values = mahalanobis_algorithm(df, params, weights)

    result = {}
    for index in range(3):
        aux = df.iloc[sorted_values.iloc[index].name].to_dict()
        aux['similarity'] = sorted_values.iloc[index][0]
        aux['similarity_index'] = int(sorted_values.iloc[index].name)

        result.update({index: aux})
        df = df[df.id != aux['id']]

    weights = {
        'value': [10.0],
        'rooms': [10.0],
        'area': [10.0],
        'bathrooms': [10.0],
        'garages': [10.0],
        'district': [30.0],
        'distance': [30.0]
    }

    sorted_values = mahalanobis_algorithm(df, params, weights)

    for index in range(3, 6):
        aux = df.iloc[sorted_values.iloc[index].name].to_dict()
        aux['similarity'] = sorted_values.iloc[index][0]
        aux['similarity_index'] = int(sorted_values.iloc[index].name)

        result.update({index: aux})

    return result


if __name__ == '__main__':
    # app.run(debug=True, host="127.0.0.1")
    app.run(debug=True, host="0.0.0.0")
