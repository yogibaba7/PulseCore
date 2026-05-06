

import sklearn
import lightgbm
import pandas as pd
import numpy as np
import joblib
import json
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import logging
import mlflow
from dotenv import load_dotenv
import os
import sys
from scipy.sparse import load_npz,csr_matrix
import yaml
from mlflow.models.signature import infer_signature



load_dotenv()

token = os.getenv("PULSECORE_PAT")
username = os.getenv("PULSECORE_USERNAME")
print(token)
print(username)

if not token:
    raise Exception("DAGSHUB_PAT not set")

# set MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token


dagshub_url = "https://dagshub.com"
repo_owner = "yogibaba7"
repo_name = "Youtube_comment_analysis"
# setup mlflow tracking 
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



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

# Load the model
def load_model(model_path: str) -> LGBMClassifier:
    logger.debug(f"loading the model from {model_path}")
    try:
        model = joblib.load(model_path)
        return model

    except Exception as e:
        logger.error(f"Unexpected error while loading model: {e}")
        return None


# make prediction
def predict(model: LGBMClassifier, x_test: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    logger.debug('Making prediction')
    try:
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)
        return y_pred, y_pred_proba
    except Exception as e:
        logger.error(f"Error while making predictions: {e}")
        return np.array([]), np.array([])
    

# scores
def store_result(path_dir: str, y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) ->dict:

    try:
        os.makedirs(path_dir,exist_ok=True)
        file_path = os.path.join(path_dir,"metrics.json")
        logger.debug('calculating results')

        # generate report dictionary
        results = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        # save json
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

        logger.debug(f"Results saved at {file_path}")

    except Exception as e:
        logger.error(f"Error while storing results: {e}")

# SaveModelInfo
def SaveModelInfo(model_uri: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'model_uri': model_uri, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


# main
def main():
    mlflow.set_experiment("DVC-Pipeline")
    with mlflow.start_run() as run:
        try:
            # data and model path
            X_test_path = 'data/interim/X_test.npz'
            X_train_path = 'data/interim/X_train.npz'
            y_test_path = 'data/interim/y_test.csv'
            y_train_path = 'data/interim/y_train.csv'
            model_path = 'models/LGBMClassifier.joblib'

            # load data
            X_test = load_data(X_test_path,'csr')
            X_train = load_data(X_train_path,'csr')
            y_test = load_data(y_test_path,'csv')
            y_train = load_data(y_train_path,'csv')

            # load model
            Model = load_model(model_path)
            
            # make prediction
            y_pred,y_pred_proba = predict(Model,X_test)

            file_path = 'reports'
            metrics = store_result(file_path, y_test, y_pred, y_pred_proba)
            logger.debug('model evaluation successfully completed')

            # set tags
            mlflow.set_tag('mlflow.runName',f"DVC-Pipeline")
            mlflow.set_tag('ExpType','MainModel')
            mlflow.set_tag('ModelType',"LGBMClassifier")

            # set description
            mlflow.set_tag('Description',f"Tuned LGBMClassifier with TFIDF Vecterization with (1,3) ngram and 10000 features and  Smote Imbalanced Data handling technique")

            # log params
            mlflow.log_param("model", "LGBMClassifier")
            mlflow.log_param("feature_engineering", "TF-IDF")
            mlflow.log_param("Ngram","(1,3)")
            mlflow.log_param("Max_features",10000)
            mlflow.log_param("Resampler Technique","Smote")

            params = LoadParams('model_building')
            # print(f"params : {params}")
            mlflow.log_params(params)

            
            mlflow.log_param("X_train_shape", X_train.shape[0])
            mlflow.log_param("X_test_shape", X_test.shape[0])
           
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("lightgbm_version", lightgbm.__version__)
            mlflow.log_param("pandas_version", pd.__version__)
            
            mlflow.log_param("classification_threshold", 0.5)
            
           

            # Log metrics 
            for metric_name, metric_value in metrics.items():
                # nested metrics
                if isinstance(metric_value, dict):
                    for name, value in metric_value.items():
                        mlflow.log_metric(
                            f"{metric_name}_{name}",
                            value
                        )
                # single metrics like accuracy
                else:
                    mlflow.log_metric(
                        metric_name,
                        metric_value
                    )
                    
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            mlflow.log_metric("model_size_mb", size_mb)
            

            # log artifacts
            mlflow.log_artifact("reports/metrics.json")
            # Log vectorizer 
            mlflow.log_artifact("models/vector.joblib", artifact_path="preprocessor")
            # Log resampler
            mlflow.log_artifact("models/resampler.joblib", artifact_path="preprocessor")


            # log model with signature and input example
            input_example = X_test[:5]
            signature = infer_signature(X_test, y_pred)
            model_info = mlflow.sklearn.log_model(
                sk_model=Model,
                artifact_path="PULSECORE_MODEL",
                signature=signature,
                input_example=input_example
            )

            print(model_info.model_uri)
            # Save model info
            SaveModelInfo(model_info.model_uri,"PULSECORE_MODEL","reports/model_info.json")

        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
