
import pandas as pd 
import numpy as np 
from lightgbm import LGBMClassifier
import logging
import os 
import sys
from scipy.sparse import load_npz
import yaml
import joblib

# configure logging
logger = logging.getLogger('model_building_logs')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LOAD PARAMS
def LoadParams(stage:str)->dict:
    try:
        with open('params.yaml','r') as f:
            file = yaml.safe_load(f)

        logger.debug('Params Loaded...')
        return file['model_building']
    except Exception as e:
        logger.error(f"Error : {e}")

# LOAD DATA
def load_data(data_path:str,data_type:str)->pd.DataFrame:
    try:
        if data_type=='csr':
            # load csr matrix
            df = load_npz(data_path)
            return df
        else:
            # load pd dataframe
            df = pd.read_csv(data_path)
            return df
        logger.debug(f"data loaded successfully from {data_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")

# fit model
def Fit(X_train,y_train,params,ModelSavedir,ModelName):
    try :
        # Create model save path
        os.makedirs(ModelSavedir,exist_ok=True)
        save_path = os.path.join(ModelSavedir,ModelName)
        # create model
        model = LGBMClassifier(**params,random_state=42)
        # train model
        model.fit(X_train,y_train.values.ravel())
        # save model
        joblib.dump(model,save_path)
        logger.debug(f"Model Trained and Saved On {save_path}")
    except Exception as e:
        logger.error(f"Error : {e}")


# main
def main():
    try:
        # load training data
        X_train = load_data('data/interim/X_train.npz','csr')
        y_train = load_data('data/interim/y_train.csv','pd.dataframe')
        print(type(X_train))
        print(type(y_train))

        # load params
        params = LoadParams("model_building")

        # train and save model
        Fit(X_train,y_train,params,ModelSavedir='models',ModelName="LGBMClassifier.joblib")

        logger.debug('Model Built...')
    
    except Exception as e:
        logger.debug('Model Building Failed')
        logger.error(f"Error : {e}")

if __name__=='__main__':
    main()