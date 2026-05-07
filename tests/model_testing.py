
import unittest
import mlflow
import os
import pandas as pd
import joblib

from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# LOAD ENV VARIABLES
# =====================================================

load_dotenv()


class BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        print("\n========== SETUP STARTED ==========")

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

        # =====================================================
        # 3. LOAD MODEL
        # =====================================================

        cls.model_name = "PULSECORE_MODEL"

        cls.model_version = cls.get_latest_model_version(
            cls.model_name
        )

        if cls.model_version is None:
            raise Exception(
                f"No staging model found for {cls.model_name}"
            )

        print(f"Model Version: {cls.model_version}")

        cls.model_uri = (
            f"models:/{cls.model_name}/{cls.model_version}"
        )

        cls.model = mlflow.pyfunc.load_model(
            cls.model_uri
        )

        print("Model Loaded Successfully")

        # =====================================================
        # 4. LOAD ARTIFACTS
        # =====================================================

        client = mlflow.MlflowClient()

        mv = client.get_model_version(
            cls.model_name,
            cls.model_version
        )

        run_id = mv.run_id

        print(f"Run ID: {run_id}")

        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=run_id
        )

        print(f"Artifacts Downloaded: {artifact_path}")

        # =====================================================
        # 5. LOAD VECTORIZER
        # =====================================================

        vector_path = os.path.join(
            artifact_path,
            "preprocessor",
            "vector.joblib"
        )

        if not os.path.exists(vector_path):
            raise FileNotFoundError(
                f"Vectorizer not found: {vector_path}"
            )

        cls.vectorizer = joblib.load(vector_path)

        print("Vectorizer Loaded")

        # =====================================================
        # 6. LOAD RESAMPLER
        # =====================================================

        resampler_path = os.path.join(
            artifact_path,
            "preprocessor",
            "resampler.joblib"
        )

        if os.path.exists(resampler_path):

            cls.resampler = joblib.load(resampler_path)

            print("Resampler Loaded")

        else:

            cls.resampler = None

            print("Resampler Not Found (Optional)")

        # =====================================================
        # 7. LOAD TEST DATA
        # =====================================================

        test_path = "data/processed/test_processed.csv"

        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"Test data not found: {test_path}"
            )

        cls.test_data = pd.read_csv(test_path)

        cls.test_data = cls.test_data.fillna("empty")

        cls.text_col = cls.test_data.columns[0]
        cls.target_col = cls.test_data.columns[1]

        print("Test Data Loaded")

        print("========== SETUP COMPLETED ==========\n")

    # =====================================================
    # GET LATEST STAGING VERSION
    # =====================================================

    @staticmethod
    def get_latest_model_version(
        model_name,
        stage="Staging"
    ):

        client = mlflow.MlflowClient()

        versions = client.get_latest_versions(
            model_name,
            stages=[stage]
        )

        if len(versions) == 0:
            return None

        return versions[0].version


# =====================================================
# FUNCTIONAL TESTS
# =====================================================

class TestFunctional(BaseTest):

    def test_model_loaded(self):

        self.assertIsNotNone(self.model)

    def test_vectorizer_loaded(self):

        self.assertIsNotNone(self.vectorizer)

    def test_prediction_output(self):

        sample_text = [
            "This product is amazing"
        ]

        X = self.vectorizer.transform(sample_text)

        # IMPORTANT:
        # Resampler should NOT be used in inference.
        # Resampling is ONLY for training.

        X = pd.DataFrame.sparse.from_spmatrix(X)

        preds = self.model.predict(X)

        self.assertEqual(len(preds), 1)

        self.assertTrue(preds[0], [0,1,2])

    def test_invalid_input(self):

        bad_input = pd.DataFrame({
            "wrong_col": ["hello"]
        })

        with self.assertRaises(Exception):

            self.model.predict(bad_input)


# =====================================================
# DATA VALIDATION TESTS
# =====================================================

class TestDataValidation(BaseTest):

    def test_no_nulls(self):

        total_nulls = (
            self.test_data.isnull().sum().sum()
        )

        self.assertEqual(total_nulls, 0)

    def test_text_column_type(self):

        first_value = (
            self.test_data[self.text_col]
            .iloc[0]
        )

        self.assertTrue(
            isinstance(first_value, str)
        )

    def test_labels_exist(self):

        labels = self.test_data[
            self.target_col
        ]

        self.assertGreater(len(labels), 0)


# =====================================================
# PERFORMANCE TESTS
# =====================================================

class TestPerformance(BaseTest):

    def test_model_performance(self):

        raw_text = self.test_data[
            self.text_col
        ]

        y_true = self.test_data[
            self.target_col
        ]

        # =====================================================
        # VECTORIZE
        # =====================================================

        X_vec = self.vectorizer.transform(
            raw_text
        )

        X_vec = pd.DataFrame.sparse.from_spmatrix(
            X_vec
        )

        # =====================================================
        # PREDICTION
        # =====================================================

        preds = self.model.predict(X_vec)

        # =====================================================
        # METRICS
        # =====================================================

        acc = accuracy_score(y_true, preds)

        precision = precision_score(
            y_true,
            preds,
            average="weighted"
        )

        recall = recall_score(
            y_true,
            preds,
            average="weighted"
        )

        f1 = f1_score(
            y_true,
            preds,
            average="weighted"
        )

        print("\n========== MODEL METRICS ==========")

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

        print("===================================\n")

        # =====================================================
        # ASSERTIONS
        # =====================================================

        self.assertGreaterEqual(acc, 0.40)

        self.assertGreaterEqual(
            precision,
            0.40
        )

        self.assertGreaterEqual(
            recall,
            0.40
        )

        self.assertGreaterEqual(
            f1,
            0.40
        )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    unittest.main(verbosity=2)