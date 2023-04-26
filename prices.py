#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from flask_cors import CORS
from xgboost import XGBRegressor
import numpy as np

model = joblib.load('regresion.pkl')

# Definición aplicación Flask
app = Flask(__name__)

# Habilitación del modelo para todas las rutas y orígenes
CORS(app)

# Definición API Flask
api = Api(
    app,
    version = '1.0',
    title = 'API PREDICCIÓN PRECIOS VEHÍCULOS USADOS',
    description = 'API que predice el precio de un vehículo usado')

ns = api.namespace('Predicción', description = 'Problema de Regresión')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'Year', 
    type = int, 
    required = True)

parser.add_argument(
    'Mileage', 
    type = int, 
    required = True)

parser.add_argument(
    'State', 
    type = int, 
    required = True)

parser.add_argument(
    'Make', 
    type=int, 
    required=True)

parser.add_argument(
    'Model', 
    type=int, 
    required=True)

resource_fields = api.model('Resource', {
    'result': fields.String,
})

# Definición de la clase para disponibilización
@ns.route('/')
class PredictApi(Resource):

    @api.doc(parser = parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": model.predict(np.array([args['Year'],
                                           args['Mileage'],
                                           args['State'],
                                           args['Make'],
                                           args['Model']]).reshape(1, -1))
        }, 200
    
if __name__ == '__main__':
    app.run(debug = True, use_reloader = False, host = '0.0.0.0', port = 5000)

