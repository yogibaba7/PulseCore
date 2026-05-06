
import pandas as pd 
import numpy as np 
import os 
import logging
import sys
import re
import yaml
from sklearn.model_selection import train_test_split


# configure logging
logger = logging.getLogger('Data_ingestion_log')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LOAD PARAMS
def load_params(module_name:str,parameter_name:str):
    with open('params.yaml','r') as f:
        file = yaml.safe_load(f)
    param = file[module_name][parameter_name]
    return param

# LOAD DATA
def load_data(data_path:str)->pd.DataFrame:
    try:
        # read data from url
        df = pd.read_csv(data_path)
        print(df['category'].value_counts())
        logger.debug(f"{df.shape[0]} rows and {df.shape[1]} columns {df.columns} loaded from data")
        logger.debug(f"data loaded successfully from {data_path}")

        return df
    except Exception as e:
        logger.error(f"error occured -> {e}")


# train test split the data.
def create_train_test_set(data:pd.DataFrame):
    try:
        # train test split
        train_data, test_data = train_test_split(
            data,
            test_size=load_params('data_ingestion', 'test_size'),
            random_state=42,
            stratify=data['category']
        )

        logger.debug(f"trainset shape {train_data.shape}")
        logger.debug(f"testingset shape {test_data.shape}")
        return train_data,test_data
    
    
    except Exception as e:
        logger.error(f"error occured -> {e}")

# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        os.makedirs(save_path,exist_ok=True)
        data.to_csv(os.path.join(save_path,save_name),index=False)
        logger.debug(f"data saved in {save_path}")


    except Exception as e:
        logger.error(f"error occured -> {e}")


# def main
def main():
    try:
        # load data
        data = load_data('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
        

        # make training and testing set
        train_data,test_data = create_train_test_set(data)


        # create save 
        save_path = os.path.join('data','raw')


        # save training data
        save_data(train_data,save_path=save_path,save_name='training_set.csv')
        

        # save testing data
        save_data(test_data,save_path=save_path,save_name='testing_set.csv')
        logger.debug('DataIngestion Completed.....')


    except Exception as e:
        logger.error(f"DataIngestion Failed")

if __name__=="__main__":
    main()












    




df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()