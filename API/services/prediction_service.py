
from API.Preprocessing import MainPreprocess
from API.Load_Model import LoadModel, LoadVector


model = LoadModel("PULSECORE_MODEL", "Production")
vector = LoadVector("PULSECORE_MODEL")



def predict_comments(comments: list[dict]):
    processed_comments = [MainPreprocess(comment['comment']) for comment in comments]

    comment_vectors = vector.transform(processed_comments)
    predictions = model.predict(comment_vectors)

    results = []
    
    for item, prediction in zip(comments, predictions):
        if prediction == 0:
            sentiment = "Neutral"
        elif prediction == 1:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        results.append({
            "comment": item['comment'],
            "published_at": item["published_at"],
            "sentiment": sentiment,
        })

    return results