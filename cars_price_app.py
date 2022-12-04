from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import pickle
from sklearn.preprocessing import StandardScaler
import re

app = FastAPI(docs_url="/docs")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

GE = 9.8
columns_with_na = pickle.load(open("columns_with_na.pkl", "rb"))
imp_median = pickle.load(open("imp_median.pkl", "rb"))
st_scal = pickle.load(open("st_scal.pkl", "rb"))
lin_reg_columns = pickle.load(open("lin_reg_columns.pkl", "rb"))
lin_reg = pickle.load(open("lin_reg.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
num_columns = pickle.load(open("num_columns.pkl", "rb"))

def predict_price(X):
    # treatment columns "mileage", "engine", "max_power" 
    X_copy = X.copy()

    X_copy.drop(columns=["selling_price"], inplace=True)

    for column in ["mileage", "engine", "max_power"]: 
        X_copy[column] = X_copy[column].str.extract('([\d.]+)', expand=True).astype(float)
    
    # treatment columns "torque" 
    X_copy['torque'] = X_copy['torque'].str.lower()
    tmp = X_copy['torque'].str.extract('([\d.,]+)[\D]*([\d.,]+)[\D]*([\d.,]*)', expand=True)
    tmp[2] = np.where(tmp[2] == "", tmp[1], tmp[2])
    for column in tmp:
        tmp[column] = tmp[column].str.replace(",","")
        tmp[column] = tmp[column].astype(float)
    tmp[0] = np.where(X_copy['torque'].str.contains("kg"), tmp[0] * GE, tmp[0])
    tmp[3] = tmp[[1, 2]].mean(axis=1)
    X_copy[['torque', 'max_torque_rpm']] = tmp[[0, 3]]

    # SimpleImputer fill nan with median
    X_copy[columns_with_na] = imp_median.transform(X_copy[columns_with_na])

    # cast to int
    for column in ["engine", "seats"]: 
        X_copy[column] = X_copy[column].astype(int)
    # scale features
    X_copy[num_columns] = st_scal.transform(X_copy[num_columns])


    y_pred = lin_reg.predict(X_copy[num_columns])

    X_copy["prediction"] = y_pred
    
    return X_copy    


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data_df = pd.json_normalize(item.dict())
    res = predict_price(data_df)
    return float(res.loc[0, "prediction"])

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    data_df = pd.DataFrame([item.dict() for item in items])
    res = predict_price(data_df)
    return res["prediction"].tolist()

