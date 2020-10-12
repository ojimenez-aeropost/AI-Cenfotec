import io

from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse, request
import pickle
import numpy as np
import random
import base64
import cv2

# variables Flask
app = Flask(__name__)
api = Api(app)

# se carga el modelo de Logistic Regression del Notebook #3
pkl_filename = "ModeloLR.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

mints_model_pkl = "mnist.pkl"
with open(mints_model_pkl, 'rb') as file:
    mnist_model = pickle.load(file)


class Predict(Resource):

    @staticmethod
    def post():
        # parametros
        parser = reqparse.RequestParser()
        parser.add_argument('petal_length')
        parser.add_argument('petal_width')
        parser.add_argument('sepal_length')
        parser.add_argument('sepal_width')

        # request para el modelo
        args = parser.parse_args()
        datos = np.fromiter(args.values(), dtype=float)
        # prediccion
        out = {'Prediccion': int(model.predict([datos])[0])}

        return out, 200

    @staticmethod
    def get():
        petal_length = float(request.args.get('pl'))
        petal_width = float(request.args.get('pw'))
        sepal_length = float(request.args.get('sl'))
        sepal_width = float(request.args.get('sw'))
        datos = [petal_length, petal_width, sepal_length, sepal_width]
        out = {'Prediccion': int(model.predict([datos])[0])}
        return out, 200

    @staticmethod
    @app.route('/predict/mnist', methods=['POST'])
    def mnist():
        parser = reqparse.RequestParser()
        parser.add_argument('img')
        args = parser.parse_args()
        base64_decoded = base64.b64decode(args.img)
        file_bytes = np.asarray(bytearray(base64_decoded), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)
        resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA).reshape(1, -1)
        out = np.asarray(mnist_model.predict(resized))
        result = {
            "digit": out.item(0)
        }
        cv2.imwrite('{}.{}.jpg'.format(out.item(0), random.random()), img)
        return jsonify(result), 200


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, port='1080')
