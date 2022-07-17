import os
import numpy as np
import pandas as pd
import config
from data.downloader import reddit
import data.articles as articles
import data.duplicates as duplicates


def get_articles_df(start_date, end_date):
    """
    obtain articles from downloadable sources.

    Parameters
    ----------
    start_date : datetime.datetime
        start date from which data is obtained
    end_date : datetime.datetime
        end date from which data is obtained

    Returns
    -------
    pandas.DataFrame
        dataframe with obtained articles
    """
    df = reddit.download_posts(start_date, end_date)
    df = articles.add_articles_to_df(df)
    df = articles.filter_invalid_articles(df)
    df = articles.filter_negative_articles(df)

    df = df.dropna(subset=["article_publish_date"])

    return df


def get_formatted_output_df(df, predictions):
    """
    Formats an article dataframe with model predicted labels

    Parameters
    ----------
    df : pandas.DataFrame
        article dataframe
    predictions : dict
        dictionary of predictions

    Returns
    -------
    pandas.DataFrame
        formatted dataframe
    """

    df["event_title"] = [
        str(f"{category} at {location}")
        for category, location in zip(
            predictions["category"], [p.name for p in predictions["location"]]
        )
    ]
    df["landslide_category"] = predictions["category"]
    df["landslide_trigger"] = predictions["trigger"]
    df["location_description"] = [p.name for p in predictions["location"]]
    df["latitude"] = [p.lat for p in predictions["location"]]
    df["longitude"] = [p.lng for p in predictions["location"]]
    df["interval_start"] = [p.interval_start for p in predictions["time"]]
    df["interval_end"] = [p.interval_end for p in predictions["time"]]
    df["location_accuracy"] = [
        str(round(p.radius)) + "km" if p.radius != np.inf else None
        for p in predictions["location"]
    ]
    df["event_date"] = [
        p.discrete_date for p in predictions["time"]
    ]
    df["event_date_accuracy"] = [
        str(round(p.confidence)) + "hours" if p.confidence != np.inf else None
        for p in predictions["time"]
    ]
    df["fatality_count"] = predictions["casualties"]

    df = df.dropna(
        subset=[
            "location_description",
            "latitude",
            "longitude",
            "interval_start",
            "interval_end",
        ]
    )

    df = df.rename(
        columns={
            "created_utc": "reddit_created_utc",
            "selftext": "reddit_text",
            "headline": "reddit_title",
            "article_title": "event_description",
        }
    )

    df = df.drop(
        columns=[
            "created",
            "score",
            "keyword",
            "text",
            "article_summary",
            "sub_text",
            "lang",
            "similarity",
        ]
    )

    df = duplicates.remove_duplicates(
        df,
        pd.read_csv(
            os.path.join(
                config.data_path, "nasa", "nasa_global_landslide_catalog_point.csv"
            )
        ),
    )

    return df
