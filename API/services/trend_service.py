
import pandas as pd


def generate_monthly_trend(results):
    df = pd.DataFrame(results)

    # Convert to datetime
    df["published_at"] = pd.to_datetime(df["published_at"])

    # Extract month
    df["month"] = df["published_at"].dt.to_period("M").astype(str)

    # Count comments by month and sentiment
    counts = (
        df.groupby(["month", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    # Convert counts to percentages
    percentages = counts.div(
        counts.sum(axis=1),
        axis=0
    ) * 100

    percentages = percentages.round(1)

    # Convert to frontend-friendly format
    trend_data = []

    for month, row in percentages.iterrows():
        trend_data.append({
            "month": month,
            "positive": row.get("Positive", 0),
            "neutral": row.get("Neutral", 0),
            "negative": row.get("Negative", 0),
        })

    return trend_data