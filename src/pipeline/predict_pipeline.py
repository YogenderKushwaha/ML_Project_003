import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path= preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.info("Error occured while Prediction Pipeline")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
        LIMIT_BAL: float,
        SEX: str,
        EDUCATION:str,
        MARRIAGE: str,
        AGE: int,
        PAY_0: str,
        BILL_AMT1:float,
        PAY_AMT1:float
        ):

        self.LIMIT_BAL = LIMIT_BAL
        self.SEX =SEX
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.AGE = AGE
        self.PAY_0 = PAY_0
        self.BILL_AMT1 = BILL_AMT1
        self.PAY_AMT1 = PAY_AMT1

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "EDUCATION": [self.EDUCATION ],
                "MARRIAGE": [self.MARRIAGE],
                "AGE": [self.AGE],
                "PAY_0": [self.PAY_0],
                "BILL_AMT1": [self.BILL_AMT1],
                "PAY_AMT1": [self.PAY_AMT1],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.info("Exception Occured in get data as dataframe part")
            raise CustomException(e, sys)



    