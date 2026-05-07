

# promote model

import mlflow
import os
from dotenv import load_dotenv

load_dotenv()



def promote_model():
    # =====================================================
    # 1. DAGSHUB / MLFLOW AUTH
    # =====================================================

    token = os.getenv("PULSECORE_PAT")
    username = os.getenv("PULSECORE_USERNAME")

    if not token:
            raise EnvironmentError(
                "PULSECORE_PAT environment variable is missing"
        )

    if not username:
            raise EnvironmentError(
                "PULSECORE_USERNAME environment variable is missing"
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    # =====================================================
    # 2. SET TRACKING URI
    # =====================================================

    dagshub_url = "https://dagshub.com"
    repo_owner = "yogibaba7"
    repo_name = "Youtube_comment_analysis"

    tracking_uri = (
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
    )

    mlflow.set_tracking_uri(tracking_uri)

    print(f"Tracking URI: {tracking_uri}")


    client = mlflow.MlflowClient()

    versions = client.get_latest_versions("PULSECORE_MODEL",stages=["Staging"])
    print(versions)
    latest_version = versions[0].version

    # archive the current production model
    prod_version = client.get_latest_versions("PULSECORE_MODEL",stages=['Production'])
    if len(prod_version)>0:
        # assuming there is only one model in production
        print("Current Production model transitioning to Archieved")
        client.transition_model_version_stage(
            name="PULSECORE_MODEL",
            version = prod_version[0].version,
            stage='Archived'
        )
    else:
        print("No Model In Production")

    # promote the new model to production
    client.transition_model_version_stage(
        name="PULSECORE_MODEL",
        version = latest_version,
        stage="Production"

    )

    print(f"Model version {latest_version} promoted to Production stage")

if __name__=="__main__":
    promote_model()