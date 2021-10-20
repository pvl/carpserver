# CARP model server

Implementation of an API server for the CARP model for text reviewing from the paper [Cut the CARP: Fishing for zero-shot story evaluation](https://arxiv.org/abs/2110.03111). Model code from this [Colab notebook](https://colab.research.google.com/drive/1nBEcy4bzM3OCFzvhX4Ii5GfSmOE-Quad) [shared by Louis Castricato](https://twitter.com/lcastricato/status/1446586951834947587).


## Download the CARP model weights

Run the following command to download the weights

```
$ cd carpserver/models
$ wget https://the-eye.eu/public/AI/CARP_L.pt
```

## Run the API server

Run the following command to start the API server

```
$ uvicorn carpserver.server:app
```

It should start a server on port 8000, the API documentation and testing page is `http://127.0.0.1:8000/docs`. The endpoint `review` takes a body with the following format:

```
{
  "text": "here the text for the story that is the subject of review",
  "reviews": [
      "This kind of drags on.",
      "This is really depressing.",
      "This is really exciting.",
      "This is boring.",
    ]
}
```

Where `reviews` is a list of reviews texts (can be anything in that list) for which the model will return scores.