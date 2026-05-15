
import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBEAPI_KEY")


def fetch_comments(video_id: str, max_comments: int = 500):
    all_comments = []
    next_page_token = None

    while len(all_comments) < max_comments:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"

        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "key": YOUTUBE_API_KEY,
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(url, params=params)
        data = response.json()

        for item in data.get("items", []):
            comment = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["textDisplay"]
               
            )
            published_at = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["publishedAt"]
            )
            all_comments.append({
                "comment": comment,
                "published_at": published_at
            })


        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return all_comments[:max_comments]



def fetch_total_comment_count(video_id: str):
    url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "statistics",
        "id": video_id,
        "key": YOUTUBE_API_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    return int(data["items"][0]["statistics"].get("commentCount", 0))