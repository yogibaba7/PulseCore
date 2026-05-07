
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field
from typing import List,Annotated

from Preprocessing import MainPreprocess
from Load_Model import LoadVector,LoadModel

class UserComment(BaseModel):
    comment : Annotated[str,Field(...,description="Comment")]

    @computed_field
    @property
    def PreprocessedComment(self)->str:
        return MainPreprocess(self.comment)
    

app = FastAPI()

# load model and vector
model = LoadModel("PULSECORE_MODEL",'Production')
vector = LoadVector("PULSECORE_MODEL")


@app.get("/")
def Home():
    return {"Message":"This API Is for SentimentAnalysis of Comments"}

@app.post("/predict")
def Predict(Comment:UserComment):
    comment = Comment.PreprocessedComment
    comment_arr = vector.transform([comment])
    prediction = model.predict(comment_arr)
    if prediction==0:
        sentiment = "Neutral"
    elif prediction==1:
        sentiment= "Positive"
    else:
        sentiment= "Negetive"
    return JSONResponse(content={"Prediction":sentiment},status_code=200)