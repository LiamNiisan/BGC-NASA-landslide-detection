import pandas as pd
from collections import defaultdict
from datetime import datetime
from dateutil import parser

from extraction.location import location


def get_nasa_db_radius(accuracy):
    if not pd.isnull(accuracy):
        if accuracy == "exact" or accuracy == "Known exactly":
            output = 0.1
        if accuracy.lower() == "unknown":
            output = 100
        if accuracy.endswith("km"):
            output = float(accuracy[:-2])
    else:
        output = None
    return output


def drop_predicted_duplicates(df):
    """Drop rows where both date, loc are empty or duplicated"""
    idxs = df.query(
        "event_date != event_date & location_description != location_description"
    ).index.to_list()
    if idxs:
        df = df.drop(idxs, axis=0)
    df = df.drop_duplicates(subset=["location_description", "event_date"])
    df = df.reset_index().drop("index", axis=1)
    return df


def get_potential_duplicates(pred, gold):
    """
    Get potential_duplicates (nasa dataset indices where the dates are duplicated)

    Parameters:
    -----------
        pred: predicted data frame containing id, locations, location, latitude,
              longitude, radius_km, interval, date, confidence columns
        gold: nasa dataset
    Returns:
        df: a data frame containing potential_duplicates column
    """
    df = pred.to_dict()
    df["potential_duplicates"] = defaultdict(str)

    gold["event_date"] = pd.to_datetime(
        gold["event_date"], format="%Y-%m-%d %H:%M", errors="coerce"
    )  # transform event_date into datetime format
    gold = gold.dropna(
        subset=["event_date"]
    )  # drop rows that are not in datetime format
    gold = gold.reset_index()  # keep the original nasa dataset index in 'index' column
    gold = gold.rename(columns={"event_date": "nasa_event_date"})

    for i in range(len(pred)):
        data = pred.iloc[[i]].merge(
            gold[["index", "nasa_event_date"]], how="cross"
        )  # index: original nasa dataset index
        if "interval_start" in pred and "interval_end" in pred:
            start = pred["interval_start"].iloc[i]
            end = pred["interval_end"].iloc[i]
            ids = data.query(
                "@start <= nasa_event_date <= @end"
            ).index.to_list()  # ids: data index
        elif (
            not pd.isnull(pred["article_publish_date"][i])
            and type(pred["article_publish_date"][i]) is str
            and pred["article_publish_date"][i] != "None"
            and pred["article_publish_date"][i] != "NaT"
        ):
            date = parser.parse(pred["article_publish_date"][i])
            ids = data.query("nasa_event_date == @date").index.to_list()
        else:
            ids = None

        if ids:
            idxs = data.iloc[ids][
                "index"
            ].to_list()  # idxs: original nasa dataset index
            df["potential_duplicates"][i] = ",".join([str(i) for i in idxs])
        else:
            df["potential_duplicates"][i] = ""
    return pd.DataFrame(df)  # "": no duplicated date or no date


def drop_nasa_duplicates(pred, gold):
    """
    Remove rows that are already in NASA dataset based on location and time

    Parameters:
    -----------
        pred: predicted data frame containing id, locations, location, latitude,
              longitude, radius_km, interval, date, potential_duplicates columns
        gold: nasa dataset
    Returns:
        df: a data frame without duplicated rows in nasa dataset
    """
    df = pred.iloc[:, :-1].copy()
    gold = gold.rename(
        columns={
            "latitude": "gold_latitude",
            "longitude": "gold_longitude",
            "location_accuracy": "gold_location_accuracy",
        }
    )
    gold = gold[
        [
            "location_description",
            "gold_location_accuracy",
            "gold_latitude",
            "gold_longitude",
        ]
    ]

    for i in range(len(pred)):
        if pred["potential_duplicates"][i] != "":
            idxs = pred["potential_duplicates"][i].split(
                ","
            )  # find index of the potential duplicates (indexes seperated by ",")
            data = pred.iloc[[i]].merge(
                gold.iloc[idxs], how="cross"
            )  # full join the current pred row with potential duplicate rows in nasa dataset
            data = data.assign(
                distance_km=data.apply(
                    lambda x: location.get_distance_lat_lng(
                        x.latitude, x.longitude, x.gold_latitude, x.gold_longitude
                    ),
                    axis=1,
                )
            )
            data = data.assign(
                gold_radius_km=data.apply(
                    lambda x: get_nasa_db_radius(x.gold_location_accuracy), axis=1
                )
            )
            data = data.assign(
                correct=data.apply(
                    lambda x: location.is_correct(
                        int(x.location_accuracy[:-2]), x.gold_radius_km, x.distance_km
                    ),
                    axis=1,
                )
            )
            data[["precision", "recall", "f1_score"]] = data.apply(
                lambda x: location.get_precision_recall_f1(
                    int(x.location_accuracy[:-2]), x.gold_radius_km, x.distance_km
                ),
                axis=1,
                result_type="expand",
            )
            if not data.query("precision > 0.5").empty:  # correct == True
                df = df.drop([i], axis=0)
    return df


def remove_duplicates(pred, gold):
    """Remove all the duplicates of prediction table and nasa dataset"""
    pred = drop_predicted_duplicates(pred)
    pred = get_potential_duplicates(pred, gold)
    pred = drop_nasa_duplicates(pred, gold)
    return pred  # remove original index: pred.iloc[:,1:]
