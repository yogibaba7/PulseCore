from API.services.analytics_service import (
    calculate_sentiment_summary,
    calculate_kpis,
)
from API.services.wordcloud_service import generate_wordcloud


# =========================================
# TEST SENTIMENT SUMMARY
# =========================================
def test_calculate_sentiment_summary():
    results = [
        {"sentiment": "Positive"},
        {"sentiment": "Positive"},
        {"sentiment": "Neutral"},
        {"sentiment": "Negative"},
    ]

    summary = calculate_sentiment_summary(results)

    assert summary["analyzed_comments"] == 4
    assert summary["positive_percent"] == 50.0
    assert summary["neutral_percent"] == 25.0
    assert summary["negative_percent"] == 25.0


# =========================================
# TEST KPIS
# =========================================
def test_calculate_kpis():
    comments = [
        "Great video",
        "Very useful tutorial"
    ]

    kpis = calculate_kpis(comments)

    assert "avg_words_per_comment" in kpis
    assert kpis["avg_words_per_comment"] > 0


# =========================================
# TEST WORD CLOUD
# =========================================
def test_generate_wordcloud():
    comments = [
        "great video",
        "great tutorial",
        "awesome video"
    ]

    top_words = generate_wordcloud(comments)

    assert isinstance(top_words, list)
    assert len(top_words) > 0