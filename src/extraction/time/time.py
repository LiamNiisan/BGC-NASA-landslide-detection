from extraction.time.helpers import (
    help_dict,
    week_days,
    times_of_day,
    months,
    months_reversed,
    seasons,
    vague_language,
)

import datetime as dt
from dateutil.relativedelta import relativedelta
import dateutil.parser as parser
from dateparser import parse
import re
from word2number import w2n
from datetime import datetime

number_of_days_in_week = 7
programmatic_number_of_days_in_week = 6
first_day = 1
hours_in_day = 24
middle_of_week = 3
end_of_week = 5
days_in_month = 30
days_in_year = 365

delta_to_subtract = relativedelta(seconds=1)
half_hour = relativedelta(minutes=30)
half_day_delta = relativedelta(hours=12)
one_day_delta = relativedelta(days=1)
end_beginning_of_week_delta = relativedelta(days=2)
weekend_delta = relativedelta(days=2)
end_of_week_delta = relativedelta(days=3)
week_delta = relativedelta(days=7)
month_delta = relativedelta(months=1)
season_delta = relativedelta(months=3)
year_delta = relativedelta(years=1)
beginning_middle_end_month_delta = relativedelta(days=10)
half_month_delta = relativedelta(days=15)
regex_time = r"[0-9]+:?[0-9]* (am|a.m.|a.m .|AM|pm|p.m.|PM)"


def words_to_number_transformer(phrase):
    try:
        return w2n.word_to_num(phrase)
    except:
        return None


def parse_date_weekday(date, day_number, modifier):
    """
    given a number of the day of the week returns the difference between the date and this particular day
    """
    parsed_date = parser.parse(date).date()
    year = parsed_date.year
    month = parsed_date.month
    day = parsed_date.day
    weekday = parsed_date.weekday() + 1
    if modifier == "last":
        delta = number_of_days_in_week + weekday - day_number
    else:
        delta = weekday - day_number
        if delta < 0:
            return None
    return delta


def seasons_interval(phrase, date_string):

    phrase = phrase.lower()
    winter = False
    spring = False
    summer = False
    autumn = False
    last_season = False

    if "winter" in phrase:
        winter = True
    if "autumn" in phrase or "fall" in phrase:
        autumn = True
    if "summer" in phrase:
        summer = True
    if "spring" in phrase:
        spring = True
    if "last" in phrase or "past" in phrase:
        last_season = True

    if last_season:
        if winter:
            date_1, date_2 = month_parsing("last " + "december", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if spring:
            date_1, date_2 = month_parsing("last " + "march", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if summer:
            date_1, date_2 = month_parsing("last " + "june", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if autumn:
            date_1, date_2 = month_parsing("last " + "september", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
    else:
        if winter:
            date_1, date_2 = month_parsing("this " + "december", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if spring:
            date_1, date_2 = month_parsing("this " + "march", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if summer:
            date_1, date_2 = month_parsing("this " + "june", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract
        if autumn:
            date_1, date_2 = month_parsing("this " + "september", date_string)
            date_2 = date_1 + season_delta - delta_to_subtract

    return date_1, date_2


def week(phrase, date_string):
    phrase = phrase.lower()
    beginning = False
    middle = False
    end = False
    last = False
    date_1 = None
    split_phrase = phrase.split(" ")
    parsed_date = parser.parse(date_string).date()
    for i, word in enumerate(split_phrase):
        if (
            word == "week"
            and i <= len(split_phrase) - 1
            and (split_phrase[i - 1] == "last" or split_phrase[i - 1] == "past")
        ):
            last = True
    if "beginning" in phrase or "early" in phrase:
        beginning = True
    if "middle" in phrase:
        middle = True
    if "end " in phrase or "late" in phrase:
        end = True
    if last:
        date_1, _, _ = last_and_this_day_calculation("last monday", date_string)
    else:
        date_1, _, _ = last_and_this_day_calculation("monday", date_string)

    date_start = parsed_date - relativedelta(days=date_1)
    date_end = date_start + week_delta - delta_to_subtract

    if beginning:
        date_end = date_start + end_beginning_of_week_delta - delta_to_subtract
    if end:
        date_start = date_start + end_of_week_delta
    if middle:
        date_start = date_start + end_beginning_of_week_delta
        date_end = date_end - end_beginning_of_week_delta
    return date_start, date_end


def weekend(phrase, date_string):
    phrase = phrase.lower()
    beginning = False
    middle = False
    end = False
    last = False
    date_1 = None
    date_2 = None

    if "beginning" in phrase or "early" in phrase:
        beginning = True
    if "middle" in phrase:
        middle = True
    if " end" in phrase or "late" in phrase:
        end = True
    if "last" in phrase or "past" in phrase:
        last = True
    if "weekend" in phrase:
        if last:
            date_1, _, _ = last_and_this_day_calculation("last saturday", date_string)
            date_2, _, _ = last_and_this_day_calculation("last sunday", date_string)
        else:
            date_1, _, _ = last_and_this_day_calculation("saturday", date_string)
            date_2, _, _ = last_and_this_day_calculation("sunday", date_string)
        parsed_date = parser.parse(date_string).date()
        date_1 = parsed_date - relativedelta(days=date_1)
        date_2 = (
            parsed_date - relativedelta(days=date_2) + one_day_delta - delta_to_subtract
        )
        if beginning:
            date_2 = date_1 + half_day_delta
        if end:
            date_1 = date_1 + 3 * half_day_delta
        if middle:
            date_1 = date_1 + half_day_delta
            date_2 = date_2 - half_day_delta
        return date_1, date_2


def time(phrase):
    around = False
    about = False
    between = False
    lst_time = []
    time_1 = None
    time_2 = None
    if "around" in phrase:
        around = True
        phrase = phrase.replace("around", "")
    if "about" in phrase:
        about = True
        phrase = phrase.replace("about", "")
    if "between" in phrase:
        between = True
    for result in re.finditer(regex_time, phrase):
        lst_time.append(result.group())

    if len(lst_time) == 1:
        time_1 = parse(lst_time[0])
    if len(lst_time) == 2:
        time_1 = parse(lst_time[0])
        time_2 = parse(lst_time[1])

    if around or about or len(lst_time) == 1:
        time_1_1 = time_1 - half_hour
        time_1_2 = time_1 + half_hour
        return time_1_1, time_1_2
    elif between and len(lst_time) == 2:
        return time_1, time_2


def month_parsing(phrase, date_string):
    phrase = phrase.lower()
    date = None
    detected_date = None
    if "month" in phrase:
        month = month_transformation(phrase, date_string)
        phrase = phrase.replace("month", month)
    phrase_list = phrase.split(" ")

    month = None
    anchor = ""
    last_year = False
    for i, word in enumerate(phrase_list):
        if word in months.keys():

            month = months[word.lower()]
            if i - 1 >= 0 and phrase_list[i - 1] == "last":
                anchor = "last"
        if word == "year" and i > 0 and phrase_list[i - 1] == "last":
            last_year = True

        try:
            if int(word):
                if len(word) <= 2:
                    date = True
                detected_date = parse(phrase)
        except:
            pass
    parsed_date = parser.parse(date_string).date()

    if month > parsed_date.month:
        anchor = "last"

    if last_year == True or (month >= parsed_date.month and anchor):
        year = parsed_date.year - 1
    else:
        year = parsed_date.year

    new_date_1 = dt.date(year, month, first_day)
    new_date_2 = new_date_1 + month_delta

    if date and detected_date:
        new_date_1 = dt.date(year, detected_date.month, detected_date.day)
        new_date_2 = new_date_1 + one_day_delta
    if "beginning" in phrase:
        new_date_2 = new_date_2 - beginning_middle_end_month_delta
    elif "middle" in phrase:
        new_date_1 = new_date_1 + beginning_middle_end_month_delta
        new_date_2 = new_date_2 - beginning_middle_end_month_delta
    elif "end" in phrase or "late" in phrase:
        new_date_1 = new_date_2 - beginning_middle_end_month_delta
    elif "late" in phrase or "later" in phrase:
        new_date_1 = new_date_1 + half_month_delta
    elif "early" in phrase or "earlier" in phrase:
        new_date_2 = new_date_1 + half_month_delta

    return new_date_1, new_date_2 - delta_to_subtract


def month_transformation(month_phrase, date_string):
    """
    given the date, identifies what month is meant in the str phrase
    """
    phrase_list = month_phrase.split(" ")
    month = None
    anchor = ""
    month_index = len(phrase_list) - 1
    for i, word in enumerate(phrase_list):
        if word == "month":
            month_index = i
        if i >= 1 and phrase_list[i - 1] == "last" and word == "month":
            anchor = "last"
    parsed_date = parser.parse(date_string).date()
    if anchor:
        month = parsed_date - month_delta
        return months_reversed[month.month]
    else:
        month = parsed_date.month
        return months_reversed[month]


def date_and_time_identifier(phrase):
    phrase = phrase.lower()
    phrase_list = phrase.split(" ")
    date = None
    date_phrase = None
    time = None
    time_phrase = None
    for word in phrase_list:
        if word in times_of_day.keys():
            time = True
            time_phrase = word
        if word in week_days.keys():
            date = True
            date_phrase = word
    return time_phrase, date_phrase


def last_and_this_day_calculation(phrase, date_string):
    """
    given a date, returns a tuple with a time delta between the str date and the one mentioned in the phrase,
    the hour for the start_date and the hour for the end_date

    """
    phrase = phrase.lower()
    time_phrase, date_phrase = date_and_time_identifier(phrase)
    days_delta, time_start, time_delta = None, None, None
    if date_phrase:
        day_number = week_days[date_phrase]
        current_weekday = parse(date_string).weekday() + 1
        if "last" in phrase or "previous" in phrase or day_number >= current_weekday:
            days_delta = parse_date_weekday(date_string, day_number, "last")
        else:
            days_delta = parse_date_weekday(date_string, day_number, "")
    if "next" in phrase:
        days_delta = None
    if time_phrase and not date_phrase:
        if "last" in phrase:
            days_delta = 1
        if "this" in phrase:
            days_delta = 0
    if time_phrase:
        time_start = times_of_day[time_phrase][0]
        time_end = times_of_day[time_phrase][1]
        if time_start > time_end:
            time_delta = hours_in_day - time_start + time_end
        else:
            time_delta = time_end - time_start

        if "early" in phrase:
            time_delta = time_delta / 2
        if "late" in phrase:
            time_delta = time_delta / 2
            time_start = time_start + time_delta

    return days_delta, time_start, time_delta


def days_time_triplet(phrase, date_string):
    days_delta, time_start, time_delta = last_and_this_day_calculation(
        phrase, date_string
    )
    event_date = None
    date_start, date_end = None, None
    publication_date = parse(date_string).date()
    if days_delta:
        event_date = parser.parse(date_string).date() - relativedelta(days=days_delta)
        date_start = event_date
        date_end = event_date + one_day_delta - delta_to_subtract
    if days_delta and time_start:
        date_start = event_date + relativedelta(hours=time_start)
        date_end = date_start + relativedelta(hours=time_delta)
    elif time_start:
        date_start = dt.date(
            publication_date.year, publication_date.month, publication_date.day
        ) + relativedelta(hours=time_start)
        date_end = date_start + relativedelta(hours=time_delta)

    return date_start, date_end


def exact_date(phrase, date_string):
    "given the date in the past in a string, returns time delta between this string and parsed date"

    parsed_phrase = parse(phrase)
    phrase_month = parsed_phrase.month
    phrase_day = parsed_phrase.day
    phrase_year = parsed_phrase.year
    parsed_date = parser.parse(date_string)
    current_year = parse("now").year

    if current_year != phrase_year:
        phrase_date = dt.date(phrase_year, phrase_month, phrase_day)
    else:
        if parsed_date.month - phrase_month <= 0:
            if (
                parsed_date.month - phrase_month == 0
                and parsed_date.day - phrase_day > 0
            ):
                phrase_date = dt.date(parsed_date.year, phrase_month, phrase_day)
            else:

                phrase_date = dt.date(parsed_date.year - 1, phrase_month, phrase_day)
        else:
            phrase_date = dt.date(parsed_date.year, phrase_month, phrase_day)

    parsed_date = dt.date(parsed_date.year, parsed_date.month, parsed_date.day)
    delta = parsed_date - phrase_date
    return delta.days


def days_count(phrase, date_string):
    def is_today():
        try:
            return parser.parse(phrase).date() == datetime.today().date()
        except:
            return False

    if is_today():
        return None

    if "yesterday" in phrase:
        return 1
    elif "today" in phrase or "tonight" in phrase:
        return 0

    unit = None
    try:
        days = exact_date(phrase, date_string)
        return days
    except:
        pass
    tokenized_phrase = phrase.split(" ")
    for i in range(len(tokenized_phrase)):
        if (
            words_to_number_transformer(tokenized_phrase[i])
            or tokenized_phrase[i] in help_dict.keys()
        ):
            if words_to_number_transformer(tokenized_phrase[i]):
                tokenized_phrase[i] = words_to_number_transformer(tokenized_phrase[i])
            else:
                tokenized_phrase[i] = help_dict[tokenized_phrase[i]]
    if "ago" in phrase or "from now" in phrase or "before" in phrase:

        if "day" in tokenized_phrase or "days" in tokenized_phrase:
            unit = first_day
        elif "week" in tokenized_phrase or "weeks" in tokenized_phrase:
            unit = number_of_days_in_week
        elif "month" in tokenized_phrase or "months" in tokenized_phrase:
            unit = days_in_month
        elif "year" in tokenized_phrase or "years" in tokenized_phrase:
            unit = days_in_year
        for item in tokenized_phrase:
            if type(item) == int:
                return unit * item
    else:
        return None


def specific_year(phrase):
    try:
        phrase_date = parse(phrase)
        date_start = dt.date(phrase_date.year, 1, 1)
        date_end = date_start + year_delta - delta_to_subtract
        return date_start, date_end
    except:
        return None, None


def between_phrase(phrase, date_string):
    phrase = phrase.lower()

    month = False
    year = False
    part_of_day = False
    weekday = False
    season = False
    month_name = None
    day_number = None

    between_regex = r"(?<=between )(.*? and .*?$)"
    date_phrases = []
    for result in re.finditer(between_regex, phrase):
        date_phrases.append(result.group())
    try:
        date_phrases = date_phrases[0].replace("and", ",").split(",")
    except:
        return None, None
    month_count = set()
    for date in date_phrases:
        try:
            if int(date):
                if len(str(int(date))) == 4:
                    year = True
                elif len(str(int(date))) <= 2:
                    day_number = True
        except:
            for month_local in months.keys():
                if month_local in date:
                    month = True
                    month_count.add(month_local)
                    month_name = month_local
            for time_local in times_of_day.keys():
                if time_local in date:
                    part_of_day = True
            for day_local in week_days.keys():
                if day_local in date:
                    weekday = True
            for season_local in seasons.keys():
                if season_local in date:
                    season = True
    date_start = None
    date_end = None
    index_to_replace = None
    if month:
        if day_number and month_name and len(month_count) == 1:
            for i, phrase in enumerate(date_phrases):
                if month_name not in phrase:
                    date_phrases[i] = phrase + " " + month_name

        if day_number is None:
            date_start, date_end = None, None
        date_start, _ = month_parsing(date_phrases[0], date_string)
        _, date_end = month_parsing(date_phrases[1], date_string)
    if part_of_day or weekday:
        date_start, _ = days_time_triplet(date_phrases[0], date_string)
        _, date_end = days_time_triplet(date_phrases[1], date_string)
    if season:
        date_start, _ = seasons_interval(date_phrases[0], date_string)
        _, date_end = seasons_interval(date_phrases[1], date_string)
    if year:
        date_start, _ = specific_year(date_phrases[0])
        _, date_end = specific_year(date_phrases[1])
    return date_start, date_end


def years_interval(phrase, date_string):
    phrase = phrase.lower()
    vague_phrase = False
    index = None
    phrase_list = phrase.split(" ")
    time_unit = None
    for i, word in enumerate(phrase_list):
        if word == "years":
            time_unit = "years"
            index = i
        elif word == "months":
            time_unit = "months"
            index = i
        elif word == "weeks":
            time_unit = "weeks"
            index = i
        elif word == "days":
            time_unit = "days"
            index = i
    phrase_before = " ".join(phrase_list[0 : i + 1])
    if (
        "several" in phrase_before
        or "couple" in phrase_before
        or "some" in phrase_before
    ):
        vague_phrase = True
    if vague_phrase:
        date_start, _ = time_interval_with_number_of_days(
            f"six {time_unit} ago", date_string
        )
        _, date_end = time_interval_with_number_of_days(
            f"three {time_unit} ago", date_string
        )
    return date_start, date_end


def time_interval_with_number_of_days(phrase, date_string):
    try:
        number_of_days_to_subtract = days_count(phrase, date_string)
        if number_of_days_to_subtract < 0:
            return None, None
        date_start = parser.parse(date_string).date() - relativedelta(
            days=number_of_days_to_subtract
        )
        date_end = date_start + one_day_delta - delta_to_subtract
        return date_start, date_end
    except:
        return None, None


def week_month(phrase, date_string):
    phrase = phrase.lower()
    last_week = False
    last_month = False
    first = False
    second = False
    third = False
    month = None
    date_beginning = None
    date_end = None

    split_phrase = phrase.split(" ")
    for i, word in enumerate(split_phrase):
        if word == "week" and i > 0:
            if split_phrase[i - 1] == "last":
                last_week = True
            elif split_phrase[i - 1] == "first":
                first = True
            elif split_phrase[i - 1] == "second":
                second = True
            elif split_phrase[i - 1] == "third":
                third = True
        if word == "week" and (
            "last" in split_phrase[i:] or "past" in split_phrase[i:]
        ):
            last_month = True
        if word in months.keys():
            month = word
    if "month" in phrase:
        month = month_transformation(phrase, date_string)

    if last_month:
        date_1, date_2 = month_parsing("last " + month, date_string)
    else:
        date_1, date_2 = month_parsing("this " + month, date_string)

    date_beginning, date_end = date_1, date_2

    if date_1.weekday() <= middle_of_week:
        partial_date_2 = date_1 + relativedelta(
            days=programmatic_number_of_days_in_week - date_1.weekday()
        )
    else:
        partial_date_2 = date_1 + relativedelta(
            days=programmatic_number_of_days_in_week
            - date_1.weekday()
            + number_of_days_in_week
        )
    if first:
        date_2 = partial_date_2

    if second:
        date_1 = partial_date_2 + one_day_delta
        date_2 = partial_date_2 + week_delta

    if third:
        date_1 = partial_date_2 + week_delta + one_day_delta
        date_2 = partial_date_2 + 2 * week_delta

    if last_week:

        if date_2.weekday() <= middle_of_week:
            date_1 = date_2 - relativedelta(
                days=(date_2.weekday() + number_of_days_in_week)
            )
        else:
            date_1 = date_2 - relativedelta(days=date_2.weekday())

    if date_1.month < months[month]:
        date_1 = date_beginning
    if date_2.month > months[month]:
        date_2 = date_end

    return date_1, date_2


def weekend_month(phrase, date_string):
    phrase = phrase.lower()

    last_weekend = False
    last_month = False
    first_weekend = False
    second_weekend = False
    third_weekend = False
    month = None

    date_beginning = None
    date_end = None

    split_phrase = phrase.split(" ")
    for i, word in enumerate(split_phrase):
        if word == "weekend" and i > 0:
            if split_phrase[i - 1] == "last":
                last_weekend = True
            elif split_phrase[i - 1] == "first":
                first_weekend = True
            elif split_phrase[i - 1] == "second":
                second_weekend = True
            elif split_phrase[i - 1] == "third":
                third_weekend = True
        if word == "weekend" and (
            "last" in split_phrase[i:] or "past" in split_phrase[i:]
        ):
            last_month = True
        if word in months.keys():
            month = word
    if "month" in phrase:
        month = month_transformation(phrase, date_string)
    if last_month:
        date_1, date_2 = month_parsing("last " + month, date_string)
    else:
        date_1, date_2 = month_parsing("this " + month, date_string)

    date_beginning, date_end = date_1, date_2
    current_weekday = date_beginning.weekday()
    last_day_weekday = date_end.weekday()

    if current_weekday < end_of_week:
        first_weekend_date = date_1 + relativedelta(days=end_of_week - current_weekday)
    else:
        first_weekend_date = date_1 + relativedelta(
            days=number_of_days_in_week + end_of_week - current_weekday
        )

    if first_weekend:
        date_1 = first_weekend_date
        date_2 = date_1 + weekend_delta - delta_to_subtract
    if second_weekend:
        date_1 = first_weekend_date + relativedelta(days=7)
        date_2 = date_1 + weekend_delta - delta_to_subtract
    if third_weekend:
        date_1 = first_weekend_date + relativedelta(days=14)
        date_2 = date_1 + weekend_delta - delta_to_subtract
    if last_weekend:
        if last_day_weekday >= end_of_week:
            date_1 = date_2 - relativedelta(days=abs(end_of_week - last_day_weekday))
            date_2 = date_1 + weekend_delta - delta_to_subtract
        else:
            date_1 = date_2 - relativedelta(days=last_day_weekday + 2)
            date_2 = date_1 + weekend_delta - delta_to_subtract

    return date_1, date_2


def specfied_time_intergation(date_start, date_end, time_start, time_end, date_string):

    if date_start and date_end:
        date_start_updated = dt.date(
            date_start.year, date_start.month, date_start.day
        ) + relativedelta(hours=time_start.hour, minutes=time_start.minute)
        date_end_updated = dt.date(
            date_end.year, date_end.month, date_end.day
        ) + relativedelta(hours=time_end.hour, minutes=time_end.minute)
    else:
        date_start = parse(date_string)
        date_start_updated = dt.date(
            date_start.year, date_start.month, date_start.day
        ) + relativedelta(hours=time_start.hour, minutes=time_start.minute)
        date_end_updated = dt.date(
            date_start.year, date_start.month, date_start.day
        ) + relativedelta(hours=time_end.hour, minutes=time_end.minute)
    return date_start_updated, date_end_updated


def exact_date_checker(phrase):
    date_definer = ["ago", "from now", "before", "yesterday", "today", "tonight"]
    if any(word in phrase for word in date_definer):
        return True
    try:
        return bool(parser.parse(phrase))
    except:
        return None


def get_discrete_date_and_confidence(date_start, date_end):
    if date_start and date_end:
        if type(date_start) == str:
            date_start = parse(date_start)
        if type(date_end) == str:
            date_end = parse(date_end)
        delta_hours = (date_end - date_start).total_seconds() / 60 / 60 / 2
        date_start += relativedelta(hours=delta_hours)

        discrete_date = str(date_start.strftime("%Y-%m-%d %H:%M"))
        confidence = round(delta_hours, 2)

        return discrete_date, confidence
    else:
        return None, None


def str_to_datetime(text):
    try:
        datetime_text = parser.parse(text)
        return datetime_text
    except parser.ParserError:
        return None


def time_date_normalization(phrase, date_string):

    phrase = phrase.replace(": ", ":").replace(" :", ":")
    if "next" in phrase:
        return None, None
    # publication_date = parse(date_string).replace(tzinfo=None)
    weekday = False
    week_in_phrase = False
    month_name = False
    month_in_phrase = False
    day_time = False
    exact_date = False
    weekend_in_phrase = False
    between_in_phrase = False
    exact_time = False
    year = False
    season = False
    number_of_years = 0
    vague_phrase = False

    date_start, date_end = None, None
    time_start, time_end = None, None

    split_phrase = phrase.split(" ")
    if re.search(regex_time, phrase):
        exact_time = True

    try:
        if exact_date_checker(phrase.lower()):
            exact_date = True
    except:
        pass

    for word in split_phrase:
        if word == "weekend":
            weekend_in_phrase = True
        elif word == "week":
            week_in_phrase = True
        elif word == "month":
            month_in_phrase = True
        elif word.lower() in times_of_day.keys():
            day_time = True
        elif word.lower() in months.keys():
            month_name = True
        elif word.lower() in week_days.keys():
            weekday = True
        elif word.lower() in seasons.keys():
            season = True
        elif "between" in word:
            between_in_phrase = True
        elif word in vague_language and "ago" in phrase:
            vague_phrase = True
        try:
            possible_year_or_date = int(word)
            if len(word) == 4:
                year = True
                number_of_years += 1
        except:
            pass

    if exact_date and not (
        vague_phrase
        or weekday
        and weekend_in_phrase
        and month_in_phrase
        and between_phrase
        and season
        and week
    ):
        date_start, date_end = time_interval_with_number_of_days(phrase, date_string)

    elif between_in_phrase and (
        month_name or month_in_phrase or year or season or day_time
    ):
        date_start, date_end = between_phrase(phrase, date_string)

    elif weekend == True and (month_in_phrase or month_name):
        date_start, date_end = weekend_month(phrase, date_string)

    elif week == True and (month_in_phrase or month_name):
        date_start, date_end = week_month(phrase, date_string)

    elif weekday or day_time:
        date_start, date_end = days_time_triplet(phrase, date_string)

    elif month_in_phrase or month_name:
        date_start, date_end = month_parsing(phrase, date_string)

    elif weekend_in_phrase and not (
        weekday
        and week_in_phrase
        and month_name
        and month_in_phrase
        and day_time
        and exact_date
        and between_phrase
        and exact_time
        and year
        and season
    ):
        date_start, date_end = weekend(phrase, date_string)

    elif week_in_phrase and not (
        weekday
        and weekend_in_phrase
        and month_name
        and month_in_phrase
        and day_time
        and exact_date
        and between_phrase
        and exact_time
        and year
        and season
    ):
        date_start, date_end = week(phrase, date_string)

    elif season and not (
        weekday
        and weekend_in_phrase
        and month_name
        and month_in_phrase
        and day_time
        and exact_date
        and between_phrase
        and exact_time
        and year
        and week
    ):
        date_start, date_end = seasons_interval(phrase, date_string)

    elif year and number_of_years == 1:
        date_start, date_end = specific_year(phrase)

    elif vague_phrase:
        date_start, date_end = years_interval(phrase, date_string)

    if exact_time:
        time_start, time_end = time(phrase)

    if weekday or day_time:
        date_start, date_end = days_time_triplet(phrase, date_string)

    if time_start and time_end:
        date_start_updated, date_end_updated = specfied_time_intergation(
            date_start, date_end, time_start, time_end, date_string
        )
        return date_start_updated, date_end_updated
    else:
        if date_start and date_end:
            try:
                date_start.hour
            except:
                #             if type(date_start) == datetime.date:
                date_start = datetime.combine(date_start, datetime.min.time())
            try:
                date_end.hour
            except:
                #             if type(date_end) == datetime.date:
                date_end = date_end + one_day_delta - delta_to_subtract
            return date_start, date_end
        else:
            return None, None
