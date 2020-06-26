# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:31:25 2020

@author: LENOVO
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_age():
    
    """Let's Predict age of a User 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: gender
        in: query
        type: number
        required: true
      - name: religion
        in: query
        type: number
        required: true
      - name: caste
        in: query
        type: number
        required: true
      - name: country
        in: query
        type: number
        required: true
      - name: mother_tongue
        in: query
        type: number
        required: true
      - name: height_cm
        in: query
        type: number
        required: true  
    responses:
        200:
            description: The output values
        
    """
    gender=request.args.get("gender")
    religion=request.args.get("religion")
    caste=request.args.get("caste")
    country=request.args.get("country")
    mother_tongue=request.args.get("mother_tongue")
    height_cm=request.args.get("height_cm")
    
    prediction=model.predict([[gender, religion, caste, country, mother_tongue, height_cm]])
    print(prediction)
    return "The estimate age is:"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_age_file():
    """Let's Predict the Age of a User 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=model.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run()
    