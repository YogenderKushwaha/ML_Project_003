import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok = True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        logging.info("Error Occured while Saving the pickle file")
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test,y_test,models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report 

    except Exception as e:
        logging.info("Getting Error on Evaluate Models")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path,"rb") as file_object:
            return pickle.laod(file_object)

    except Exception as e:
        logging.info("Error Occured while Load object")
        raise CustomException(e, sys)



def custom_encoder(x):
    try:  
        mapping = {'Pay Duly': -1, 'Pay delay for 1 month': 1, 'Pay delay for 2 months': 2, 'Pay delay for 3 months': 3,
         'Pay delay for 4 months': 4,'Pay delay for 5 months': 5, 'Pay delay for 6 months': 6, 'Pay delay for 7 months': 7,
          'Pay delay for 8 months': 8, 'Pay delay for 9 months or more': 9}
        return x.map(mapping)

    except:
        logging.info("Error Occured in custom encoder")
        raise CustomException(e, sys)

