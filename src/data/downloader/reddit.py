"""
This script is used to download data from Reddit
that contains relevant keywords about landslides.
The data is then filtered to make sure to only keep relevant information.
"""
import os
import json
import pandas as pd
import datetime as dt
from psaw import PushshiftAPI

import config

with open(os.path.join(config.model_path, "landslide_lexicon.json")) as f:
    lexicon = json.load(f)

POSITIVE_LEXICON = lexicon["positive"]
START_EPOCH = int(dt.datetime(2019, 1, 1).timestamp())


def download_posts(start_date=None, end_date=None):
    """
    Queries all the posts that that have words in the positive
    lexicon. The posts are then downloaded in a json file.
    """
    config.logger.info("downloading Reddit posts...")
    if start_date:
        config.logger.info(f"specified start date : {start_date.date()}")
    if end_date:
        config.logger.info(f"specified end date : {end_date.date()}")

    api = PushshiftAPI()
    all_results = []

    if start_date:
        after = int(start_date.timestamp())
    else:
        after = START_EPOCH

    before = int(end_date.timestamp())

    for keyword in POSITIVE_LEXICON:
        config.logger.info(f"querying examples with keyword : {keyword}...")
        keyword_results = list(
            api.search_submissions(
                after=after,
                before=before,
                q=keyword,
                subreddit="",
                filter=[
                    "url",
                    "title",
                    "selftext",
                    "created_utc",
                    "subreddit",
                    "full_link",
                    "score",
                ],
                limit=100000,
            )
        )
        for post in keyword_results:
            post.d_["keyword"] = keyword
        all_results.extend([post.d_ for post in keyword_results])
        config.logger.info(f"keyword {keyword} downloaded with {len(keyword_results)}.")
    df = pd.DataFrame([])
    if all_results:
        df = get_filtered_df(all_results)
    df.to_csv(
        os.path.join(
            config.data_path,
            "reddit",
            f"reddit-{str(start_date.date()) if start_date else ''}-{str(end_date.date()) if end_date else ''}.csv",
        ),
        index=False,
    )
    config.logger.info("done")
    return df


def get_filtered_df(all_results):
    """
    Filters all the results from reddit by making sure there
    is no duplicate and makes sure that all the keywords are
    present in the post.

    Returns
    -------
    pandas.DataFrame()
        Dataframe with all the filtered results
    """
    config.logger.info("filtering posts...")
    df = pd.DataFrame(all_results)
    # Removing anything posts that have the same url
    df = df.drop_duplicates(subset=["full_link"], ignore_index=True)
    df = df.drop_duplicates(subset=["title", "url"], ignore_index=True)
    # Removing posts that don't have the keywords in the lexicon in:
    # 1. The post title
    # 2. The post text
    # 3. The post url
    df = df[
        df["title"].map(lambda x: any([keyword in x for keyword in POSITIVE_LEXICON]))
        | df["selftext"].map(
            lambda x: any(
                [keyword in x for keyword in POSITIVE_LEXICON if type(x) is str]
            )
        )
        | df["url"].map(
            lambda x: any(
                [keyword in x for keyword in POSITIVE_LEXICON if type(x) is str]
            )
        )
    ]
    df["text"] = df["title"] + "\n" + df["selftext"]
    df = df.rename(
        columns={"full_link": "reddit_url", "url": "source_link", "title": "headline"}
    )
    return df
