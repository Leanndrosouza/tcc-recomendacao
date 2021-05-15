from flask import Flask, request
from utils import mahalanobis_algorithm, valid_params, prepare_params, load_properties, handle_dataframe_values

app = Flask(__name__)


@app.route('/')
def hello():
    error = valid_params(request)

    if error != None:
        return error, 404

    params, error = prepare_params(request)

    df = load_properties()

    df, params = handle_dataframe_values(df, params)

    weights = {
        'value': [3.0],
        'rooms': [1.0],
        'area': [1.0],
        'bathrooms': [1.0],
        'garages': [1.0],
        'district': [1.0],
        'distance': [1.0]
    }

    sorted_values = mahalanobis_algorithm(df, params, weights)

    result = {}
    for index in range(6):
        aux = df.iloc[sorted_values.iloc[index].name].to_dict()

        result.update({index: aux})

    return result


if __name__ == '__main__':
    app.run(debug=True)
