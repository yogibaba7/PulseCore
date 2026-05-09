def calculate_sentiment_summary(results):
    positive = sum(1 for r in results if r["sentiment"] == "Positive")
    neutral = sum(1 for r in results if r["sentiment"] == "Neutral")
    negative = sum(1 for r in results if r["sentiment"] == "Negative")

    total = len(results)

    return {
        "analyzed_comments": total,
        "positive_percent": round((positive / total) * 100, 1),
        "neutral_percent": round((neutral / total) * 100, 1),
        "negative_percent": round((negative / total) * 100, 1),
    }



def calculate_kpis(comments):
    total_words = sum(len(comment.split()) for comment in comments)

    avg_words_per_comment = round(total_words / len(comments), 1)

    return {
        "avg_words_per_comment": avg_words_per_comment,
    }