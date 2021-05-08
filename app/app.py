from flask import Flask, request
from utils import mahalanobis_algorithm, valid_params, prepare_params, load_properties

app = Flask(__name__)


@app.route('/')
def hello():
    error = valid_params(request)

    if error != None:
        return error, 404

    params, error = prepare_params(request)

    df = load_properties()

    weights = {'value': [1.0], 'rooms': [3.0], 'area': [0.5]}

    sorted_values = mahalanobis_algorithm(df, params, weights)

    result = {}
    for index in range(6):
        aux = df.iloc[sorted_values.iloc[index].name].to_dict()
        
        result.update({index: aux})

    return result


if __name__ == '__main__':
    app.run(debug=True)
