import json
import pickle
import re
import ssl

import nltk
import numpy as np
import pandas as pd

from fastapi import FastAPI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from pydantic import BaseModel

# fix for nltk error: SSL Certificate verify failed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')


app = FastAPI()


class Item(BaseModel):
    title: str


def text_processing(text: str) -> str:
    """ process text for the trained model"""
    text = text.lower()  # lower casing
    text = re.sub('-', '- ', text)  # add space after -
    text = re.sub(r'[^\w\s+\']+', '', text)  # remove special characters
    text = re.sub('/', '', text)  # remove / character
    text = ' '.join(text.split())  # remove extra spacing
    text = ''.join(i for i in text if not i.isdigit())  # remove numbers

    text_token = word_tokenize(text)  # nltk tokenization by word

    extra_list = ['nan', 'x', 'cm', 'neu', 'm', 'mm', 'gr', 'de', 's', 'inkl', 'l', 'top', 'original',
                  'zustand', 'ab', 'gb', 'à', 'd', 'pro', 'kg', 'lieferung']  # Additional stopwords
    stop_words = set(stopwords.words('german'))  # german stop words
    english_words = set(stopwords.words('english'))  # some titles are in English
    french_words = set(stopwords.words('french'))  # some titles are in French
    stop_words.update(extra_list)
    stop_words.update(english_words)
    stop_words.update(french_words)

    # tokenization
    test_text_token = [word for word in text_token if word not in stop_words]
    # Stemming
    stemmer = SnowballStemmer("german")
    text_stemmed = ' '.join(stemmer.stem(y) for y in test_text_token)

    return text_stemmed


def import_tags() -> dict:
    """ import tags as dict"""
    tf = open("./tags.json", "r")
    tags = json.load(tf)
    return tags


@app.post("/classify")
async def hello(data: Item):
    with open("./final_model.model", "rb") as file:
        model = pickle.load(file)

    tags = import_tags()  # import tags as dict

    processed_text = text_processing(data.title)  # preprocess text
    model.predict(pd.Series(processed_text))  # make predictions

    # extract probabilities for each tag
    probabilities = model.predict_proba(pd.Series(processed_text))
    order = np.argsort(probabilities, axis=1)
    # extract top 3 classes
    top_3_classes = model.classes_[order[:, -3:]]
    # extract top 3 classes probabilities
    top_3_probabilities = (probabilities[np.repeat(np.arange(order.shape[0]), 3), order[:, -3:].flatten()].
                           reshape(order.shape[0], 3))

    # final response
    response_dict = {"title": data.title,
                     "top_3_results": [
                         {"product_type": tags[str(top_3_classes[0][2])],
                          "score": round(top_3_probabilities[-1][-1], 4)},
                         {"product_type": tags[str(top_3_classes[0][1])],
                          "score": round(top_3_probabilities[-1][-2], 4)},
                         {"product_type": tags[str(top_3_classes[0][0])],
                          "score": round(top_3_probabilities[-1][-3], 4)}],
                     "product_type": tags[str(top_3_classes[0][2])]}

    return response_dict
