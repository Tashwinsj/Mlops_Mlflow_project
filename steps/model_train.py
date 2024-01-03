import logging 
import pandas as pd 
from zenml import step 
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin 
import mlflow  

from zenml.client import Client 

experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker = experiment_tracker.name)
def train_model(  
    x_train : pd.DataFrame , 
    x_test : pd.DataFrame ,
    y_train  :pd.DataFrame , 
    y_test : pd.DataFrame ,
     
) -> RegressorMixin : 
    
    try : 
        model = "None"
        if True : 
            model = LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train(x_train , y_train)
            return trained_model  
        else : 
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e : 
        logging.error("Error in training model: {} ".format(e))
        raise e 