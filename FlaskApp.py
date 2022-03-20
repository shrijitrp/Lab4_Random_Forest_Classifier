# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 11:45:03 2022

@author: noopa
"""


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request,jsonify, render_template

app=Flask(__name__)
pickle_in = open("RandomForestClf.pkl","rb")
clf=pickle.load(pickle_in)
y_map = {0: 'Bream',
 1: 'Parkki',
 2: 'Perch',
 3: 'Pike',
 4: 'Roach',
 5: 'Smelt',
 6: 'Whitefish'}

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = clf.predict(final_features)

    
    return render_template('index.html', prediction_text='The fish is of {} species'.format(prediction))
    
    


if __name__=='__main__':
    app.run()