from flask import Flask, request
from utils import mahalanobis_algorithm, valid_params

app = Flask(__name__)


@app.route('/')
def hello():
    error = valid_params(request)

    if error != None:
        return error, 404

    # params = prepare_params(request)
    return request.query_string.decode()


if __name__ == '__main__':
    app.run(debug=True)
