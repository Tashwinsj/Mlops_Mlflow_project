from telnetlib import LFLOW
from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__" :
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="/Users/tashwinsj/Desktop/mlops/customer_satisfaction/data/olist_customers_dataset.csv")
    
#mlflow ui --backend-store-uri   "file:/Users/tashwinsj/Library/Application Support/zenml/local_stores/704d2f81-dd97-4c58-a411-b5e6aad96e65/mlruns"