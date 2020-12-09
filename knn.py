import numpy as np
import pandas as pd
import math
import time
from sklearn import metrics
from statistics import mean

def Closest(values,train,k):
    distances = np.sqrt(((values - train[:, np.newaxis])**2).sum(axis = 2))
    distances = distances[~np.isnan(distances).any(axis = 1)].ravel()
    return np.argsort(distances)[:k]

def KNN(Training_data,Testing_data,class_var,num_neighb):
    Training_data = Training_data.sample(frac = 1)
    Training_features = Training_data.loc[:,Training_data.columns != class_var]
    Training_features = Training_features.values
    Testing_features = Testing_data.loc[:,Testing_data.columns != class_var]
    Testing_features = Testing_features.values
    predictions = []
    start_time = time.time()
    Results = []
    for x in range(0,len(Testing_features)):
        TE_value = np.asarray(Testing_features[x])
        closest_k = Closest(values = TE_value,train = Training_features,k = num_neighb).ravel()
        closest_classes = Training_data.iloc[closest_k][class_var].tolist()
        predictions.append(max(set(closest_classes),key = closest_classes.count))
    Results.append(time.time() - start_time)
    Results.append(predictions)
    return Results
