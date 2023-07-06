from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET',"POST"])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
          
            LIMIT_BAL= request.form.get('LIMIT_BAL'),
            SEX= request.form.get('SEX'),
            EDUCATION= request.form.get('EDUCATION'),
            MARRIAGE= request.form.get('MARRIAGE'),
            AGE= request.form.get('AGE'),
            PAY_0= float(request.form.get('PAY_0')),
            BILL_AMT1= float(request.form.get('BILL_AMT1')),
            PAY_AMT1= float(request.form.get('PAY_AMT1')),

        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(final_new_data)
        if results[0] == 1:
            result = "Yes"
        else:
            result = "No"
        return render_template('results.html',final_result= result)
        

if __name__ == "__main__":
    app.run(host= '0.0.0.0',debug = True)