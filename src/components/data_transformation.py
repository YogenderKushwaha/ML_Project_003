import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, custom_encoder
from sklearn.model_selection import GridSearchCV

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
        This function is responsible for the data transformation
        '''
        try:
            numerical_columns = ["LIMIT_BAL","AGE","BILL_AMT1","PAY_AMT1"]
            categorical_columns1 = ["SEX","MARRIAGE"]
            categorical_columns2 = ["EDUCATION"]
            categorical_columns3 = ["PAY_0"]

            # Define the custom ranking for each ordinal variable
            education_categories = ["graduate school", "university", "high school","others"]

            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline1 = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one _hot_encoding",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean = False))
                ]
            )

            cat_pipeline2 = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one _hot_encoding",OrdinalEncoder(categories=[education_categories])),
                    ("scaler",StandardScaler(with_mean = False))
                ]
            )

            cat_pipeline3 = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('encoder', FunctionTransformer(custom_encoder)),
                    ("scaler",StandardScaler(with_mean = False))
                ]
            )

            logging.info(f"Categorical Columns1:{categorical_columns1} ")
            logging.info(f"Categorical Columns2:{categorical_columns2} ")
            logging.info(f"Numerical Columns: {numerical_columns} ")
            

            preprocessor = ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ('cat_pipelines1',cat_pipeline1,categorical_columns1),
                ('cat_pipelines2',cat_pipeline2,categorical_columns2),
                ('cat_pipeline3',cat_pipeline3,categorical_columns3)
                ]
            )

            return preprocessor 



        except Exception as e:
            logging.info("Error Occured in Data Tranformation Part")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "default.payment.next.month"
            columns_to_drop = ['ID','default.payment.next.month','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                       'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
            numerical_columns = ["LIMIT_BAL","BILL_AMT1","PAY_AMT1"]

            input_feature_train_df = train_df.drop(columns=columns_to_drop, axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[columns_to_drop], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Concatanating datasets with numpy")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_obj(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            


        except Exception as e:
            logging.info("Error Occured in Initiate data transformation")
            raise CustomException(e, sys)
