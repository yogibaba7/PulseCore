from fastapi.testclient import TestClient
from unittest.mock import patch

from API.app import app

client = TestClient(app)


# =========================================
# TEST HOME ROUTE
# =========================================
def test_home_route():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "PulseCore API"
    }


# =========================================
# TEST /predict
# =========================================
@patch("API.app.predict_comments")
def test_predict_route(mock_predict_comments):
    mock_predict_comments.return_value = [
        {
            "comment": "Great video",
            "sentiment": "Positive"
        }
    ]

    payload = {
        "comments": [
            "Great video"
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "results" in data
    assert data["results"][0]["sentiment"] == "Positive"


# =========================================
# TEST /analyze-video
# =========================================
@patch("API.app.fetch_comments")
@patch("API.app.fetch_total_comment_count")
@patch("API.app.predict_comments")
@patch("API.app.calculate_sentiment_summary")
@patch("API.app.calculate_kpis")
@patch("API.app.generate_wordcloud")
@patch("API.app.generate_monthly_trend")
def test_analyze_video_route(
    mock_generate_monthly_trend,
    mock_generate_wordcloud,
    mock_calculate_kpis,
    mock_calculate_summary,
    mock_predict_comments,
    mock_fetch_total_comment_count,
    mock_fetch_comments,
):
    # Mock service outputs
    mock_fetch_comments.return_value = [
        {"comment": "Great video", "published_at": "2025-01-01T00:00:00Z"}
    ]

    mock_fetch_total_comment_count.return_value = 5000

    mock_predict_comments.return_value = [
        {
            "comment": "Great video",
            "published_at": "2025-01-01T00:00:00Z",
            "sentiment": "Positive",
        }
    ]

    mock_calculate_summary.return_value = {
        "analyzed_comments": 1,
        "positive_percent": 100.0,
        "neutral_percent": 0.0,
        "negative_percent": 0.0,
    }

    mock_calculate_kpis.return_value = {
        "avg_words_per_comment": 2.0
    }

    mock_generate_wordcloud.return_value = [
        ["great", 10],
        ["video", 8],
    ]

    mock_generate_monthly_trend.return_value = [
        {
            "month": "2025-01",
            "positive": 100.0,
            "neutral": 0.0,
            "negative": 0.0,
        }
    ]

    payload = {
        "VedioId": "abc123"
    }

    response = client.post("/analyze-video", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["total_comments"] == 5000
    assert data["positive_percent"] == 100.0
    assert data["avg_words_per_comment"] == 2.0
    assert "top_words" in data
    assert "trend_data" in data
    assert "results" in data


# =========================================
# VALIDATION TEST
# =========================================
def test_analyze_video_validation_error():
    # Missing video_id
    response = client.post("/analyze-video", json={})

    assert response.status_code == 422