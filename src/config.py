import os
import json
import logging
from datetime import datetime
import dateutil.parser as parser

main_path = os.path.join(
    os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__))),
    os.pardir,
)
model_path = os.path.join(main_path, "models")
data_path = os.path.join(main_path, "data")
user_path = os.path.join(main_path, "user")

config_file_path = os.path.join(user_path, "config.json")
history_file_path = os.path.join(user_path, "history.log")

logger = logging
logger.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(user_path, "run.log")),
        logging.StreamHandler()
    ]
)


def get_interval():
    """
    Reads config file to extract appropriate
    date interval to extract articles from.

    Returns
    -------
    tuple
        (start_date, end_date)
    """

    with open(config_file_path) as f:
        config = json.load(f)
    if config["interval"]["default"] == "yes":
        log = []
        with open(history_file_path) as f:
            for line in f:
                log.append(parser.parse(line.strip().split("\t")[2]))
        if len(log) == 0:
            start_date = datetime(2019, 1, 1)
        else:
            log.sort()
            start_date = log[-1]

        end_date = datetime.now()

    else:
        if config["interval"]["start"]["date"]:
            start_date = parser.parse(config["interval"]["start"]["date"])
        else:
            start_date = datetime(2019, 1, 1)

        if config["interval"]["end"]["now"] == "yes":
            end_date = datetime.now()
        elif config["interval"]["end"]["date"]:
            end_date = parser.parse(config["interval"]["end"]["date"])
        else:
            end_date = datetime.now()

    with open(history_file_path, "a+") as f:
        f.write("\t".join([str(datetime.now()), str(start_date), str(end_date)]) + "\n")

    return start_date, end_date


def is_running_baseline():
    """
    Looks in the config file for if the users wants to run 
    the baseline pipeline.

    Returns
    -------
    bool
        True if "yes", False if "no"
    """    
    with open(config_file_path) as f:
        config = json.load(f)

    if config["model"]["baseline"] == "yes":
        return True
    else:
        return False

def is_running_multitask():
    """
    Looks in the config file for if the users wants to run 
    the multitask pipeline.

    Returns
    -------
    bool
        True if "yes", False if "no"
    """    
    with open(config_file_path) as f:
        config = json.load(f)

    if config["model"]["multitask"] == "yes":
        return True
    else:
        return False
