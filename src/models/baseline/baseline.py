import os
import pickle
import numpy as np
import config
from joblib import Parallel, delayed
from nltk.tokenize.regexp import RegexpTokenizer

from extraction.time import time
from extraction.casualties import casualties
from extraction.time.landslide_event_time import LandslideEventTime
from extraction.location.landslide_event_location import LandslideEventLocation

from models.baseline import ner


TOKENIZER = RegexpTokenizer("\w+|\$[\d\.]+|\S+")


def extract_casualties(text):
    """
    Rule based method to extract casualties if a certain text contains a
    token number about casualties.

    Parameters
    ----------
    text : str


    Returns
    -------
    str
        number of casualties if any
    """
    tokens = TOKENIZER.tokenize(text)
    for i, token in enumerate(tokens):
        if casualties.is_num(token):
            if i + 4 > len(tokens):
                return ""
            if (
                tokens[i + 1].lower() == "dead"
                or tokens[i + 1].lower() == "died"
                or tokens[i + 1].lower() == "killed"
                or tokens[i + 1].lower() == "buried"
                or tokens[i + 2].lower() == "dead"
                or tokens[i + 2].lower() == "died"
                or tokens[i + 2].lower() == "killed"
                or tokens[i + 2].lower() == "buried"
                or tokens[i + 3].lower() == "dead"
                or tokens[i + 3].lower() == "died"
                or tokens[i + 3].lower() == "killed"
                or tokens[i + 3].lower() == "buried"
            ):
                return casualties.format_num(token)
    return ""


def is_time_sentence_invalid(row):
    if row["dates"] and len(row["dates"]) > 3:
        return True
    else:
        return False


def is_location_sentence_invalid(row):
    if row["locations"] and len(row["locations"]) > 3:
        return True
    else:
        return False


def predict_categories(texts):
    """
    Predicts landslide categories with a logistic regression model.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of categories
    """
    with open(os.path.join(config.model_path, "category.model"), "rb") as f:
        model = pickle.load(f)

    categories = model.predict(texts)

    return categories


def predict_triggers(texts):
    """
    Predicts landslide triggers with a logistic regression model.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of triggers
    """
    with open(os.path.join(config.model_path, "trigger.model"), "rb") as f:
        model = pickle.load(f)

    triggers = model.predict(texts)

    return triggers


def predict_casualties(texts):
    """
    Predicts casualties with a rule based method.

    Parameters
    ----------
    texts : list(str)
        list of strings to predict

    Returns
    -------
    list(str)
        list of casualties
    """
    casualties = [extract_casualties(text) for text in texts]

    return casualties


def predict_datetimes(sentence_df, publication_dates):
    predicted_event_times = []
    id2idx = dict()
    for i, id in enumerate(sentence_df.groupby("id")["id"].size().index):
        id2idx[id] = i
        predicted_event_times.append(LandslideEventTime([], ""))

    with open(os.path.join(config.model_path, "date_time.model"), "rb") as f:
        model = pickle.load(f)

    time_probs = model.predict_proba(sentence_df)[:, 1]

    sentence_df["time_sentence_is_positive_confidence"] = time_probs
    sentence_df = (
        sentence_df[sentence_df.apply(is_time_sentence_invalid, axis=1)]
        .copy()
        .reset_index(drop=True)
    )
    sentence_df = sentence_df.iloc[
        sentence_df.groupby("id")["time_sentence_is_positive_confidence"].idxmax()
    ].copy()

    for idx in range(sentence_df.shape[0]):
        phrases = sentence_df["dates"].iloc[idx].split("|")
        publication_date = publication_dates[id2idx[sentence_df["id"].iloc[idx]]]
        predicted_event_times[id2idx[sentence_df["id"].iloc[idx]]] = LandslideEventTime(
            phrases, publication_date
        )

    return predicted_event_times


def predict_locations(sentence_df):
    predicted_event_locations = []
    id2idx = dict()
    for i, id in enumerate(sentence_df.groupby("id")["id"].size().index):
        id2idx[id] = i
        predicted_event_locations.append(LandslideEventLocation([]))

    with open(os.path.join(config.model_path, "location.model"), "rb") as f:
        model = pickle.load(f)

    location_probs = model.predict_proba(sentence_df["text"])[:, 1]

    sentence_df["location_sentence_is_positive_confidence"] = location_probs
    sentence_df = (
        sentence_df[sentence_df.apply(is_location_sentence_invalid, axis=1)]
        .copy()
        .reset_index(drop=True)
    )
    sentence_df = sentence_df.iloc[
        sentence_df.groupby("id")["location_sentence_is_positive_confidence"].idxmax()
    ].copy()

    locations_candidates = sentence_df["locations"].to_numpy()
    extracted_event_locations = Parallel(n_jobs=-1, verbose=1)(
        delayed(LandslideEventLocation)(locations.split("|"))
        for locations in locations_candidates
    )

    for id, event_location in zip(
        sentence_df["id"].to_numpy(), extracted_event_locations
    ):
        predicted_event_locations[id2idx[id]] = event_location

    return predicted_event_locations


def predict(article_df):
    sentence_df = ner.get_NER_sentences(article_df)

    articles = article_df["article_text"].to_numpy().tolist()
    publication_dates = article_df["article_publish_date"].astype(str).to_numpy()
    publication_dates = list(map(time.str_to_datetime, publication_dates))

    event_locations = predict_locations(sentence_df)
    event_times = predict_datetimes(sentence_df, publication_dates)
    event_casualties = predict_casualties(articles)
    categories = predict_categories(articles)
    triggers = predict_triggers(articles)

    return {
        "location": event_locations,
        "time": event_times,
        "casualties": event_casualties,
        "category": categories,
        "trigger": triggers,
    }
