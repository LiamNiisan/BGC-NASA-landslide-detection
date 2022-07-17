import os
import re
import json
import pickle
import nltk
import fasttext
import numpy as np
import pandas as pd
from newspaper import Article
from urllib.parse import urlparse
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize.regexp import RegexpTokenizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc, cossim, any2sparse

import config


with open(os.path.join(config.model_path, "landslide_lexicon.json")) as f:
    lexicon = json.load(f)

POSITIVE_LEXICON = lexicon["positive"]
NEGATIVE_LEXICON = lexicon["negative"]


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPS = set(stopwords.words("english"))
TOKENIZER = RegexpTokenizer("\w+|\$[\d\.]+|\S+")

regex = r"[^0-9A-Za-z]"
URL_SPLIT_REGEX = re.compile(regex)

# Filtering threshold used to isolate positive articles
FILTERING_THRESHOLD = 0.08


def is_landslide_keyword_in_article(article):
    """
    Looks for positive lexicon words inside the article.

    Parameters
    ----------
    article : pandas.core.series.Series or dict
        article row

    Returns
    -------
    bool
        True if it has keyword, False if not
    """
    return any([keyword in article["article_text"] for keyword in POSITIVE_LEXICON])


def is_no_not_landslide_keyword_in_article(article):
    """
    Checks if words from the negative landslide lexicon aren't in the article.

    Parameters
    ----------
    article : pandas.core.series.Series or dict
        article row

    Returns
    -------
    bool
        True if it has keyword, False if not
    """
    return not any([keyword in article["article_text"] for keyword in NEGATIVE_LEXICON])


def is_n_landslide_keyword_in_article(article, n=3):
    """
    Checks if an n number of positive lexicon keywords are in the arrticle or not.

    Parameters
    ----------
    article : pandas.core.series.Series or dict
        article row
    n : int, optional
        number of required matches, by default 3

    Returns
    -------
    bool
        True if it has keyword, False if not
    """
    return (
        sum([article["text"].lower().count(keyword) for keyword in POSITIVE_LEXICON])
        >= n
    )


def filter_articles_by_lang(df, lang="en"):
    """
    Only returns articles that have the lang parameter language.

    Parameters
    ----------
    df : Pandas.DataFrame
        dataframe containing a text column
    lang : str, optional
        language code to filter, by default "en"

    Returns
    -------
    Pandas.DataFrame
        filtered dataframe
    """
    lang_model = fasttext.load_model(os.path.join(config.model_path, "lid.176.bin"))
    pred, prob = lang_model.predict(
        df["text"].str.replace("\n", " ").to_numpy().tolist()
    )
    df["lang"] = np.array(pred).squeeze(axis=-1)
    df["lang"] = df["lang"].str.replace("__label__", "")

    return df[df["lang"] == lang]


def filter_invalid_articles(df):
    """
    Filters articles by removing duplicate, checking for lexicon keywords
    and checking for language.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to filter

    Returns
    -------
    pandas.DataFrame
        filtered articles
    """
    df = df.dropna(subset=["article_title", "article_text"]).copy()
    df["sub_text"] = (
        df["article_text"].str.slice(stop=500).str.lower().str.replace("\n", " ")
    )
    df = df.drop_duplicates(subset=["sub_text"]).copy()
    df["text"] = df.apply(
        lambda x: x["article_title"] + "\n" + x["article_text"], axis=1
    )
    df = df[df.apply(is_landslide_keyword_in_article, axis=1)]
    df = df[df.apply(is_no_not_landslide_keyword_in_article, axis=1)]
    df = filter_articles_by_lang(df)
    return df


def preprocess_url(url_str, regex=URL_SPLIT_REGEX):
    """
    Extracts keywords from url string

    Parameters
    ----------
    url_str : str
        url to extract from
    re.Pattern : , optional
        compiled regex, by default URL_SPLIT_REGEX

    Returns
    -------
    str
        keywords separated by a space
    """
    url_keywords = []
    url_parsed = urlparse(url_str)
    url_keywords.extend(
        [
            keyword
            for keyword in regex.split(url_parsed.netloc)
            if keyword != ""
            and not keyword.isnumeric()
            and keyword != "com"
            and keyword != "www"
        ]
    )
    url_keywords.extend(
        [
            keyword
            for keyword in regex.split(url_parsed.path)
            if keyword != "" and not keyword.isnumeric() and keyword != "html"
        ]
    )
    return " ".join(url_keywords)


def get_article(url):
    """
    Downloads article from url.

    Parameters
    ----------
    url : str
        url to fetch article from

    Returns
    -------
    dict
        extracted information from url,
        fields are empty if nothing is extracted.
    """
    article = Article(url, fetch_images=False)
    if article.is_valid_url():
        try:
            article.download()
            article.parse()
            return {
                "url": url,
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "summary": article.summary,
                "text": article.text,
            }
        except:
            pass
    return {
        "url": url,
        "title": "",
        "authors": [],
        "publish_date": None,
        "summary": "",
        "text": "",
    }


def add_articles_to_df(df):
    """
    For a dataframe that contains urls, columns containing extracted article
    information will be added.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe containing the urls

    Returns
    -------
    pandas.DataFrame
        dataframe with the added extracted articles
    """
    df = df.sort_values(by="score", ascending=False).drop_duplicates(["source_link"])

    urls = df["source_link"].to_numpy()
    articles = Parallel(n_jobs=-1, verbose=1)(delayed(get_article)(url) for url in urls)
    articles_df = pd.DataFrame(articles)

    added_articles_df = pd.merge(
        df,
        articles_df.rename(
            columns={
                "url": "source_link",
                "title": "article_title",
                "authors": "article_authors",
                "publish_date": "article_publish_date",
                "summary": "article_summary",
                "text": "article_text",
            }
        ).drop_duplicates(["source_link"]),
        how="left",
        on="source_link",
    )

    return added_articles_df


def preprocess(sent):
    """
    Preprocess a string by tokenizing it and filtering it's tokens.

    Parameters
    ----------
    sent : str
        str to tokenize and filter

    Returns
    -------
    list
        list of tokens
    """
    tokens = TOKENIZER.tokenize(sent)
    return [
        token.lower()
        for token in tokens
        if len(token) > 2 and token.isalpha() and token.lower() not in STOPS
    ]


def vectorizer(text, model, dct):
    """
    Vectorize a string to a TF-IDF vector.

    Parameters
    ----------
    text : str
        string to vectorize
    model : gensim.models.tfidfmodel.TfidfModel
        TF-IDF model
    dct : gensim.corpora.dictionary.Dictionary
        gensim dictionary

    Returns
    -------
    list(tuple)
        vectorized string
    """
    return model[dct.doc2bow(preprocess(text))]


def get_article_similarity_score(article, tfidf_landslide_vector, model, dct):
    """
    Get article similarity score with vector passed in the parameter.

    Parameters
    ----------
    article : pandas.core.series.Series or dict
        article row from pandas dataframe.
    tfidf_landslide_vector : list(tuple)
        vector to compare text to.
    model : gensim.models.tfidfmodel.TfidfModel
        TF-IDF model
    dct : gensim.corpora.dictionary.Dictionary
        gensim dictionary

    Returns
    -------
    float
        similarity score
    """
    article_vector = vectorizer(article["text"], model, dct)
    return cossim(tfidf_landslide_vector, article_vector)


def filter_negative_articles(df):
    """
    Uses TF-IDF similarity to only keep articles related
    to landslides.

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to filter.

    Returns
    -------
    pandas.DataFrame
        filtered dataframe.
    """
    dictionary = Dictionary.load_from_text(
        os.path.join(config.model_path, "gensim_dictionary.txt")
    )
    tfidf_model = TfidfModel.load(
        os.path.join(config.model_path, "landslide_tfidf.model")
    )
    with open(os.path.join(config.model_path, "landslide_tfidf.vec"), "rb") as f:
        tfidf_vector = pickle.load(f)

    df["similarity"] = df.apply(
        get_article_similarity_score,
        tfidf_landslide_vector=tfidf_vector,
        model=tfidf_model,
        dct=dictionary,
        axis=1,
    )

    return df[df["similarity"] > FILTERING_THRESHOLD]
