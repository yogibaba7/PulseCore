
from pydantic import BaseModel, Field
from typing import Annotated, List


class UserComment(BaseModel):
    comments: Annotated[List[str], Field(..., description="Comments")]


class VideoRequest(BaseModel):
    VedioId: Annotated[str, Field(..., description="YouTube Video ID")]