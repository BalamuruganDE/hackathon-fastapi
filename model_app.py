# creating api endpoints using fastapi

#load libraires


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

class input(BaseModel):
    Gender                  :   object
    Age                     :   int
    Driving_License         :   int
    Region_Code             :   float
    Previously_Insured      :   int
    Vehicle_Age             :   object
    Vehicle_Damage          :   object
    Annual_Premium          :   float
    Policy_Sales_Channel    :   float
    Vintage                 :   int
    

class output(BaseModel):
    Response                :   object

@app.post("/predict")
def predict(data:input)->output:
    X_input = pd.DataFrame([[
        data.Gender,
        data.Age,
        data.Driving_License,
        data.Region_Code,
        data.Previously_Insured,
        data.Vehicle_Age,
        data.Vehicle_Damage,
        data.Annual_Premium,
        data.Policy_Sales_Channel,
        data.Vintage
    ]])

    X_input.columns=['Gender', 'Age', 'Driving_License', 'Region_Code',
       'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
       'Policy_Sales_Channel', 'Vintage']
    
    # model = joblib.load('../cross-sell-pred-pkl.gz')
    model = joblib.load('cross-sell-pred-xgb-pkl.gz')
    prediction = model.predict(X_input)
    
    if prediction == 0:
        op="Customer is NOT INTERESTED  in buying Vehicle Insurance"
        return output(Response=op)
    else:
        op="Customer is INTERESTED  in buying Vehicle Insurance"
        return output(Response=op)

   



'''


id                      :   int
Gender                  :   object
Age                     :   int
Driving_License         :   int
Region_Code             :   float
Previously_Insured      :   int
Vehicle_Age             :   object
Vehicle_Damage          :   object
Annual_Premium          :   float
Policy_Sales_Channel    :   float
Vintage                 :   int
Response                :   int

id                        int64
Gender                   object
Age                       int64
Driving_License           int64
Region_Code             float64
Previously_Insured        int64
Vehicle_Age              object
Vehicle_Damage           object
Annual_Premium          float64
Policy_Sales_Channel    float64
Vintage                   int64
Response                  int64


{
  "Gender": "Male",
  "Age": 44,
  "Driving_License": 1,
  "Region_Code": 28.0,
  "Previously_Insured": 0,
  "Vehicle_Age": "> 2 Years",
  "Vehicle_Damage": "Yes",
  "Annual_Premium": 40454,
  "Policy_Sales_Channel": 26,
  "Vintage": 217
}

'''