# this file loads model from model registry that is in production

import mlflow 
import mlflow.pyfunc
import json
import joblib
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def SetupDagshub():
     # setup dagshub credentials for mlflow tracking
    token = os.getenv("PULSECORE_PAT")
    username = os.getenv("PULSECORE_USERNAME")
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # uri
    dagshub_url = "https://dagshub.com"
    repo_owner = "yogibaba7"
    repo_name = "Youtube_comment_analysis"
    

    uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(uri)


def LoadModel(ModelName:str,ModelStage:str)->mlflow.pyfunc:

    SetupDagshub()

    # prepare model uri
    model_uri = model_uri = f"models:/{ModelName}/{ModelStage}"

    # load model 
    model = mlflow.pyfunc.load_model(model_uri)

    return model


# load Vector
def LoadVector(ModelName:str):
    SetupDagshub()

    client = mlflow.MlflowClient()

    model_name = ModelName
    # latest production version
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])

    run_id = latest_versions[0].run_id
    

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id)

    # load vectorizer
    vec_path = os.path.join(local_path, "preprocessor/vector.joblib")

    vectorizer = joblib.load(vec_path)

    return vectorizer















