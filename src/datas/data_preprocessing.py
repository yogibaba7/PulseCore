
import pandas as pd 
import numpy as np 
import os 
import logging
import sys
import re
import yaml
import nltk
from nltk.corpus import stopwords
# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# configure logging
logger = logging.getLogger('Data_preprocessing_log')
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


# BASIC PREPROCESSING
def Basic_Preprocessing(data:pd.DataFrame)->pd.DataFrame:
    try:
        # drop null values
        data = data.dropna()
        
        # drop rows where comment length is less then 2
        data[data['clean_comment'].apply(lambda x: len(x))<=2]

        # remove urls
        broken_url_pattern =r'https?\s+(www\s+)?(?:[a-zA-Z0-9\-]+\s+){1,3}(com|org|net|gov|edu|nic|tumblr)'
        # Remove broken URLs
        data['clean_comment'] = data['clean_comment'].apply(lambda x: re.sub(broken_url_pattern, '', x))

        # replace new line characters
        data['clean_comment'] = data['clean_comment'].str.replace('\n', ' ', regex=True)
        data['clean_comment'] = data['clean_comment'].str.replace('\t', ' ', regex=True)

        # Remove non-English characters from the 'clean_comment' column , Keeping only standard English letters, digits, and common punctuation
        data['clean_comment'] = data['clean_comment'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))

        data['clean_comment'] = data['clean_comment'].apply(lambda x : x.strip())

        # drop duplicate
        data.drop_duplicates(inplace=True)

        data['category'] = data['category'].map({0:0,1:1,-1:2})

        logger.debug(f"Basic preprocessing applied now the shape of data is {data.shape}")
        
        return data
    
    except Exception as e:
        logger.error(f"error occured -> {e}")


# remove stop words
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
        return text
    except Exception as e:
        logger.error(f"Error -> {e}")
        return text
    
# lemmatizations
def lemmatizor(text):
    try:
        lemmatizor = WordNetLemmatizer()
        text = " ".join([lemmatizor.lemmatize(word) for word in text.split()])
        return text
    except Exception as e:
        logger.error(f"Error -> {e}")
        return text



# PREPROCESSING
def MainPreprocessing(data:pd.DataFrame):
    try:
        data['clean_comment'] = data['clean_comment'].apply(remove_stop_words)
        data['clean_comment'] = data['clean_comment'].apply(lemmatizor)
        logger.debug('Preprocessing applied')
        return data
    
    except Exception as e:
        logger.debug("Preprocessing Failed..")
        logger.error(f"Error -> {e}")
        return data


# save data 
def save_data(data:pd.DataFrame,save_path:str,save_name:str):
    try:
        os.makedirs(save_path,exist_ok=True)
        data.to_csv(os.path.join(save_path,save_name),index=False)

        logger.debug(f"data saved in {save_path}")
    except Exception as e:
        logger.error(f"error occured -> {e}")


# main
def main():
    # load training data
    trainset_path = "data/raw/training_set.csv"
    train_set = load_data(trainset_path)
    
    # load testing data
    testset_path = "data/raw/testing_set.csv"
    test_set = load_data(testset_path)

    # Apply Basic Preprocessing
    train_set = Basic_Preprocessing(train_set)
    test_set = Basic_Preprocessing(test_set)

    # apply preprocessing on training data
    train_set = MainPreprocessing(train_set)
    # apply preprocessing on testing data
    test_set = MainPreprocessing(test_set)

    # create save path
    save_path = os.path.join('data','processed')

    # save processed training and testing data
    save_data(train_set,save_path=save_path,save_name='train_processed.csv')
    save_data(test_set,save_path=save_path,save_name='test_processed.csv')
    


if __name__=="__main__":
    main()