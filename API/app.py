from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import UserComment, VideoRequest

from services.youtube_service import (
    fetch_comments,
    fetch_total_comment_count,
)
from services.prediction_service import predict_comments
from services.analytics_service import (
    calculate_sentiment_summary,
    calculate_kpis,
)
from services.wordcloud_service import generate_wordcloud
from services.trend_service import generate_monthly_trend

from Preprocessing import MainPreprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "PulseCore API"}


@app.post("/predict")
def predict(data: UserComment):
    results = predict_comments(data.comments)
    return {"results": results}


@app.post("/analyze-video")
def analyze_video(data: VideoRequest):
    comments = fetch_comments(data.VedioId)
    total_comments = fetch_total_comment_count(data.VedioId)
    processed_comments = [MainPreprocess(comment['comment']) for comment in comments]

    results = predict_comments(comments)

    summary = calculate_sentiment_summary(results)
    kpis = calculate_kpis(processed_comments)

    
    top_words = generate_wordcloud(processed_comments)
    
    trend_data = generate_monthly_trend(results)


    return {
        "total_comments": total_comments,
        "results": results[:5],
        "top_words": top_words,
        "trend_data": trend_data,
        **summary,
        **kpis,
    }