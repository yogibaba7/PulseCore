
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, computed_field, Field
from typing import Annotated,List

import requests
from collections import Counter

from Preprocessing import MainPreprocess
from Load_Model import LoadVector, LoadModel
from dotenv import load_dotenv
import os
import time

load_dotenv()

yt_key = os.getenv("YOUTUBEAPI_KEY")


# Create model input  schema
class UserComment(BaseModel):

    comments: Annotated[List[str], Field(..., description="Comment")]


# vedio request schema
class VedioRequest(BaseModel):

    VedioId: Annotated[str, Field(...,description="VedioId")]




# Create FastAPI app
app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# Load ML model and vectorizer
model = LoadModel("PULSECORE_MODEL", 'Production')
vector = LoadVector("PULSECORE_MODEL")

# Fetch comments from YouTube API
def FetchComments(video_id: str,
                  max_comments: int = 200):

    # Store comments
    all_comments = []

    # Pagination token
    next_page_token = None


    while len(all_comments) < max_comments:

        # YouTube API URL
        url = (

            "https://www.googleapis.com/youtube/v3/commentThreads"
        )


        # API parameters
        params = {

            "part": "snippet",

            "videoId": video_id,

            "maxResults": 50,

            "key": yt_key
        }


        # Add pagination token
        if next_page_token:

            params["pageToken"] = next_page_token


        # API request
        response = requests.get(

            url,

            params=params
        )


        # Convert to JSON
        data = response.json()


        # Extract comments
        for item in data.get("items", []):

            comment = (

                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["textDisplay"]
            )

            all_comments.append(comment)


        # Get next page token
        next_page_token = data.get(
            "nextPageToken"
        )


        # Stop if no more pages
        if not next_page_token:

            break


    return all_comments[:max_comments]

# Fetch total comment count
def FetchTotalCommentCount(video_id: str):

    url = (

        "https://www.googleapis.com/youtube/v3/videos"
    )


    params = {

        "part": "statistics",

        "id": video_id,

        "key": yt_key
    }


    response = requests.get(
        url,
        params=params
    )


    data = response.json()


    total_comments = (

        data["items"][0]
        ["statistics"]
        ["commentCount"]
    )


    return total_comments


# Home route
@app.get("/")
def Home():

    return {
        "Message": "This API Is for SentimentAnalysis of Comments"
    }


# Prediction route
@app.post("/predict")
def Predict(Data: UserComment):

    results = []

    for comment in Data.comments:

        # preprocess single string
        processed_comment = MainPreprocess(comment)

        # vector transform
        comment_arr = vector.transform(
            [processed_comment]
        )

        # predict
        prediction = model.predict(comment_arr)[0]

        # sentiment mapping
        if prediction == 0:

            sentiment = "Neutral"

        elif prediction == 1:

            sentiment = "Positive"

        else:

            sentiment = "Negative"


        results.append({

            "comment": comment,

            "sentiment": sentiment
        })


    return JSONResponse(
        content={
            "results": results
        },
        status_code=200
    )


@app.post("/analyze-video")
def AnalyzeVideo(data: VedioRequest):

    # Fetch comments
    comments = FetchComments(
        data.VedioId
    )


    # Fetch total comment count
    total_comments = FetchTotalCommentCount(
        data.VedioId
    )


    start_time = time.perf_counter()
    
    # =====================================
    # PREPROCESS ALL COMMENTS
    # =====================================

    processed_comments = [

        MainPreprocess(comment)

        for comment in comments
    ]



    # =====================================
    # VECTORIZER TRANSFORM (ONCE)
    # =====================================

    comment_vectors = vector.transform(
        processed_comments
    )



    # =====================================
    # PREDICT ALL COMMENTS (ONCE)
    # =====================================

    predictions = model.predict(
        comment_vectors
    )



    # =====================================
    # STORE RESULTS
    # =====================================

    results = []


    # Sentiment counters
    positive = 0
    negative = 0
    neutral = 0



    # =====================================
    # LOOP THROUGH PREDICTIONS
    # =====================================

    for comment, prediction in zip(
        comments,
        predictions
    ):

        # Convert prediction
        if prediction == 0:

            sentiment = "Neutral"

            neutral += 1

        elif prediction == 1:

            sentiment = "Positive"

            positive += 1

        else:

            sentiment = "Negative"

            negative += 1


        # Save result
        results.append({

            "comment": comment,

            "sentiment": sentiment
        })

    end_time = time.perf_counter()
    inference_time = round(
    end_time - start_time,
    2)

    print(
    f"Inference Time: {inference_time} seconds")
    # Calculate percentages
    total = len(comments)


    positive_percent = round(
        (positive / total) * 100,
        1
    )

    negative_percent = round(
        (negative / total) * 100,
        1
    )

    neutral_percent = round(
        (neutral / total) * 100,
        1
    )

    # =====================================
    # AVERAGE WORDS PER COMMENT
    # =====================================

    total_words = sum(

        len(comment.split())

        for comment in comments
    )


    avg_words_per_comment = round(

        total_words / len(comments),

        1
    )

    # =====================================
    # WORD CLOUD DATA
    # =====================================
    custom_stopwords = {

        "this",
        "that",
        "with",
        "from",
        "video",
        "have",
        "they",
        "your",
        "just",
        "about"
    }

    # Combine all comments
    all_text = " ".join(comments)


    # Split into words
    words = all_text.lower().split()


    # Remove short words
    words = [

        word

        for word in words

        if len(word) > 3 and word not in custom_stopwords
    ]


    # Count frequency
    word_counts = Counter(words)


    # Get top 30 words
    top_words = word_counts.most_common(50)
    print(top_words)
    return JSONResponse(

        content={
            "top_words": top_words,

            "avg_words_per_comment":
                avg_words_per_comment,

            "total_comments":
                total_comments,

            "analyzed_comments":
                total,

            "positive_percent":
                positive_percent,

            "neutral_percent":
                neutral_percent,

            "negative_percent":
                negative_percent,

            "results":
                results[:5]
        },

        status_code=200
    )