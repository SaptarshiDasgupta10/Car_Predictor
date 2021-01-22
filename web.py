from flask import Flask,render_template,request,redirect
import sklearn
import numpy as np
import pandas as pd

import pickle

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel1.pkl','rb'))
car = pd.read_csv('cleaned_car_1.csv')

@app.route('/',methods = ['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car.name.unique())
    year = sorted(car.year.unique(),reverse = True)
    fuel_type = car.fuel_type.unique()

    companies.insert(0, 'Select Company')

    return render_template('index.html',companies = companies,car_models=car_models,years = year,fuel_type = fuel_type)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')

    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    return str(prediction[0])




if __name__ == "__main__":
    app.run(debug = True)
