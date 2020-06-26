import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def home():
    return '<h1> API works </h1>'

@app.route('/predict')
def predict_age():
    prediction = model.predict([[request.args['gender'],
                                 request.args['religion'],
                                 request.args['caste'],
                                 request.args['country'],
                                 request.args['mother_tongue'],
                                 request.args['height_cm']]])
    return str(prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)   