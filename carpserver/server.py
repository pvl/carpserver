from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from carpserver.carp import TextReviewer

app = FastAPI()

reviewer_model = TextReviewer()


class ReviewItem(BaseModel):
    text: str
    reviews: List[str]


@app.post("/review")
async def review(item: ReviewItem):
    res = reviewer_model.review(item.text, item.reviews, pairs=False)
    return res
