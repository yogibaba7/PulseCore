
import pandas as pd 
import numpy as np 
import logging
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from imblearn.over_sampling import SMOTE
from scipy.sparse import save_npz,csr_matrix

# configure logging
logger = logging.getLogger('feature_engineering_logs')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# LOAD DATA
def load_data(data_path:str)->pd.DataFrame:
    try:
        # read data from url
        df = pd.read_csv(data_path)
        
        logger.debug(f"{df.shape[0]} rows and {df.shape[1]} columns {df.columns} loaded from data")
        logger.debug(f"data loaded successfully from {data_path}")

        return df
    except Exception as e:
        logger.error(f"error occured -> {e}")

# train and save vectorization model
def TrainVector(data:pd.DataFrame,save_path:str)->pd.DataFrame:
    try:
        tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=10000)
        data = tfidf.fit(data['clean_comment'])
        logger.debug('Vector Trained..')
        os.makedirs(save_path,exist_ok=True)
        joblib.dump(tfidf,os.path.join(save_path,'vector.joblib'))
        logger.debug(f"Vector Saved on {save_path}")
        logger.debug('TrainVector Completed..')
        return data
    except Exception as e:
        logger.error(f"Error : {e}")



# apply Resampling on training data and save resampler
def ApplyResampling(X_train:pd.DataFrame,y_train:pd.DataFrame,Savesamplerdir:str,Samplername:str)->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        os.makedirs(Savesamplerdir,exist_ok=True)
        save_path = os.path.join(Savesamplerdir,Samplername)
        smt = SMOTE(random_state=42)
        X_train,y_train = smt.fit_resample(X_train,y_train)

        joblib.dump(smt,save_path)

        logger.debug(f"Model Saved on {save_path} ")
        logger.debug('Resampling Applied')
        return X_train,y_train
    except Exception as e:
        logger.error(f"Error : {e}")

# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        
        os.makedirs(save_path,exist_ok=True)
        if isinstance(data, pd.Series):
            data.to_csv(os.path.join(save_path, save_name), index=False)
        elif isinstance(data, csr_matrix):
            save_npz(os.path.join(save_path, save_name.replace('.csv', '.npz')), data)
       
        logger.debug(f"data saved in {save_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")

# main
def main():
    # load training data
    train_set = load_data('data/processed/train_processed.csv')
    
    # load testing data
    test_set = load_data('data/processed/test_processed.csv')
    

    # drop null
    train_set = train_set.dropna()
    test_set = test_set.dropna()

    print(train_set.columns)
    # Crearte inputs and outputs
    X_train = train_set.drop(columns=['category'])
    y_train = train_set['category']

    X_test = test_set.drop(columns=['category'])
    y_test = test_set['category']


    # vectorsave path
    vectorsave_path = os.path.join('models')

    # Trainvector and save 
    vector = TrainVector(X_train,vectorsave_path)

    # transform train and test
    X_train = vector.transform(X_train['clean_comment'])
    X_test = vector.transform(X_test['clean_comment'])

    print(type(X_train))
    print(type(y_train))


    # apply resampling only on training data 
    X_train,y_train = ApplyResampling(X_train,y_train,Savesamplerdir="models",Samplername="resampler.joblib")



    # create save path
    save_path = os.path.join('data','interim')
    # save processed training data
    save_data(X_train,save_path=save_path,save_name='X_train.csv')
    save_data(y_train,save_path=save_path,save_name='y_train.csv')
    
    # save processed testing data
    save_data(X_test,save_path=save_path,save_name='X_test.csv')
    save_data(y_test,save_path=save_path,save_name='y_test.csv')

if __name__=="__main__":
    main()