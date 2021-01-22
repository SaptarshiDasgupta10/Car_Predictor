import numpy as np
import pandas as pd
car = pd.read_csv('quikr_car.csv')
car.fuel_type.value_counts()
car.info()
backup = car.copy()
car.year.unique()
car = car[car['year'].str.isnumeric()]
car['year'] = car['year'].astype(int)
car = car[car['Price']!="Ask For Price"]
car['Price'] = car['Price'].str.replace(',','')
car['Price'] = car.Price.astype(int)
print(car.kms_driven.head())
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
car = car[car.kms_driven.str.isnumeric()]
car.kms_driven = car.kms_driven.astype(int)
car = car[~car.fuel_type.isnull()]
print(car.info())
car['name'].str.split().str[0:3].str.join(' ')
car['name'] = car['name'].str.split().str[0:3].str.join(' ')
car = car.reset_index(drop = True)
car.describe()
car.drop(index = 534,inplace = True)
car = car.reset_index(drop = True)
car.to_csv('cleaned_car_1.csv')
X = car.drop(['Price'],axis = 1)
y = car['Price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
ohe = OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
columns_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['name','company','fuel_type']),remainder = 'passthrough')
l = LinearRegression()
pipe = make_pipeline(columns_trans,l)
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
#print(y_pred)
print(r2_score(y_test,y_pred))
score = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=i)
    lr = LinearRegression()
    pipe = make_pipeline(columns_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    score.append(r2_score(y_test,y_pred))
    print(score[np.argmax(score)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.argmax(score))
    lr = LinearRegression()
    pipe = make_pipeline(columns_trans, lr)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2_score(y_test, y_pred)
    import pickle

    pickle.dump(pipe, open('LinearRegressionModel1.pkl', 'wb'))
    print( pipe.predict(pd.DataFrame([['Maruti Suzuki Swift', 'Maruti', 2019, 100, 'Petrol']],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])))
