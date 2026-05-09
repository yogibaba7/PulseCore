
from Preprocessing import MainPreprocess
from Load_Model import LoadModel, LoadVector


model = LoadModel("PULSECORE_MODEL", "Production")
vector = LoadVector("PULSECORE_MODEL")



def predict_comments(comments: list[str]):
    processed_comments = [MainPreprocess(comment) for comment in comments]

    comment_vectors = vector.transform(processed_comments)
    predictions = model.predict(comment_vectors)

    results = []

    for comment, prediction in zip(comments, predictions):
        if prediction == 0:
            sentiment = "Neutral"
        elif prediction == 1:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        results.append({
            "comment": comment,
            "sentiment": sentiment,
        })

    return results