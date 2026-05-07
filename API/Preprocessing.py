import regex as re
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


# BASIC PREPROCESSING
def Basic_Preprocessing(text:str)->str:
    try:
        # lower
        text = text.lower()
        # remove urls
        broken_url_pattern =r'https?\s+(www\s+)?(?:[a-zA-Z0-9\-]+\s+){1,3}(com|org|net|gov|edu|nic|tumblr)'
        # Remove broken URLs
        text = re.sub(broken_url_pattern, '', text)

        # replace new line characters
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        # Remove non-English characters from the 'clean_comment' column , Keeping only standard English letters, digits, and common punctuation
        text = re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(text))

        text = text.strip()

        return text
    
    except Exception as e:
        print(f"error occured -> {e}")


# remove stop words
def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
        return text
    except Exception as e:
        print(f"Error -> {e}")
        return text
    
# lemmatizations
def lemmatizor(text):
    try:
        lemmatizor = WordNetLemmatizer()
        text = " ".join([lemmatizor.lemmatize(word) for word in text.split()])
        return text
    except Exception as e:
        print(f"Error -> {e}")
        return text


def MainPreprocess(text:str)-> str:
    print('apply basic preprocessing')
    text = Basic_Preprocessing(text)
    print("Removing Stop word")
    text = remove_stop_words(text)
    print("Apply Lemmatization")
    text = lemmatizor(text)
    return text