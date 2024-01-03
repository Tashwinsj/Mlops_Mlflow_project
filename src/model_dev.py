import logging 
from sklearn.linear_model import LinearRegression
from abc import ABC , abstractmethod

class Model(ABC): 
    @abstractmethod 
    def train(self , x_train  , y_train) :
        pass 



class LinearRegressionModel(Model) :
    def train(self, x_train, y_train ,**kwargs) :
        try: 
            reg = LinearRegression(**kwargs) 
            reg.fit(x_train ,y_train) 
            logging.info("Model training compleated")
            return reg 
        except Exception as e : 
            logging.error("Error in training model : {}".format(e))
            raise(e)
    
