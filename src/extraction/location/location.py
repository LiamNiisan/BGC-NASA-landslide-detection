import math
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import geocoder
import geopy.distance


def get_distance(p1, p2):
    """Get the geographical distance between two points"""
    if p1 and p2:
        return round(geopy.distance.geodesic(p1, p2).km, 3)
    else:
        return None


def get_distance_lat_lng(p1_lat, p1_lng, p2_lat, p2_lng):
    if pd.isnull(p1_lat):
        return None
    else:
        return get_distance((p1_lat, p1_lng), (p2_lat, p2_lng))


def get_radius(p1, p2):
    """Get the radius of a region"""
    if p1 and p2:
        return round(geopy.distance.geodesic(p1, p2).km, 3) / 2
    else:
        return None


def get_lat_lng_radius(location_name):
    """Get the latitude, longitude and radius or a region"""
    lat, lng, radius = None, None, None
    geocoded = geocoder.arcgis(location_name).json
    if geocoded:
        geoloc = geocoded["bbox"]
        lat, lng = geocoded["lat"], geocoded["lng"]
        ne, sw = geoloc["northeast"], geoloc["southwest"]
        radius = get_radius(ne, sw)

    return (lat, lng, radius)


def get_centroid(points):
    """Get the centroid of a group of points"""
    lats = [p[0] for p in points]
    lngs = [p[1] for p in points]

    return (np.mean(lats), np.mean(lngs))

def get_outlier_idx(centroid, points):
    """
    Parameters:
        centroid: a tuple of centroid;
        points: a list of tuples
    Return:
        the index of the point that should be removed
    """
    dists = [get_distance(centroid, point) for point in points]
    return dists.index(max(dists))


def get_smallest_region_idx(locs):
    """
    Parameters
    ----------
    locs : list of dictionary
        a list of dictionary containing latitude, longitude,
        northeast point, southwest point for all the location
        entities in the positive sentence

    Returns
    ----------
        an integer indicating the index of the location entity
        that has the smallest region
    """
    dists = [get_distance(loc["northeast"], loc["southwest"]) for loc in locs]
    return dists.index(min(dists))


def get_precision_recall_f1(r_pred, r_gold, d):
    """Get location evaluation metrics based on intersected area"""
    if r_pred >= r_gold:
        r1 = r_pred
        r2 = r_gold
    else:
        r1 = r_gold
        r2 = r_pred

    try:
        if d >= r1 + r2:
            a_intersection = 0
        elif d <= r1 - r2:
            a_intersection = math.pi * (r2**2)
        else:
            d1 = (r1**2 - r2**2 + d**2) / (2 * d)
            d2 = d - d1
            a1 = (r1**2) * math.acos(d1 / r1) - d1 * math.sqrt(r1**2 - d1**2)
            a2 = (r2**2) * math.acos(d2 / r2) - d2 * math.sqrt(r2**2 - d2**2)
            a_intersection = a1 + a2

        a_pred = math.pi * (r_pred**2)
        a_gold = math.pi * (r_gold**2)

        precision = a_intersection / a_pred
        recall = a_intersection / a_gold
    except ZeroDivisionError:
        precision = 0
        recall = 0

    if precision != 0 and recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    return precision, recall, f1


def is_correct(r_pred, r_gold, d):
    if d <= r_pred + r_gold:
        correct = True
    else:
        correct = False
    return correct

