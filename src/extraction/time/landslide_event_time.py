import numpy as np
from datetime import datetime
from extraction.time import time


class LandslideEventTime:
    def __init__(self, time_phrases, publication_date):
        self.interval_start = None
        self.interval_end = None
        self.discrete_date = None
        self.confidence = np.inf

        if type(publication_date) is datetime and time_phrases:
            for phrase in time_phrases:
                interval = time.time_date_normalization(phrase, str(publication_date))

                if not interval:
                    continue

                discrete_date, confidence = time.get_discrete_date_and_confidence(
                    interval[0], interval[1]
                )

                if confidence and confidence < self.confidence:
                    self.interval_start = interval[0]
                    self.interval_end = interval[1]
                    self.discrete_date = discrete_date
                    self.confidence = np.abs(confidence)
