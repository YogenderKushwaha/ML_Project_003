import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training & test input data")
            X_train, y_train, X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("splitted the data into training & testing")
            models = {
                "Logistic Regression": LogisticRegression(),
                "RandomForest Classifier": RandomForestClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "Support Vector Classifier": SVC(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "XGB Classifier": XGBClassifier()
            }
            params={
                "RandomForest Classifier":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Classifier":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Logistic Regression":{},
                "Support Vector Classifier": {},
                
                "XGB Classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost Classifier":{
                    'depth': [2,5,8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [30, 50, 100,200,400]
                },
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test, y_test= y_test, models = models,param = params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both training and testing dataset")

            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            f1_score = f1_score(y_test,predicted)
            return f1_score



        except Exception as e:
            logging.info("Error Occured in initiate model trainer")
            raise CustomException(e, sys)
