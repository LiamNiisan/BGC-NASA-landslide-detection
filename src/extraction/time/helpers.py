help_dict = {
    'a': 1
}

week_days = {
    'monday': 1,
    'tuesday' : 2 ,
    'wednesday' : 3 ,
    'thursday' : 4 ,
    'friday' : 5 ,
    'saturday' : 6 ,
    'sunday' : 7
}
weekdays_reversed = {v: k for k, v in week_days.items()}
times_of_day = {
    'morning':(5, 11),
    'afternoon': (12, 17),
    'evening': (18, 22),
    'night': (23, 4),
    'lunchtime': (12, 15),
    'noon': (12, 17),
    'midnight': (24, 2)
}
months = {'january': 1,
         'february': 2,
         'march': 3,
         'april': 4,
         'may': 5,
         'june': 6,
         'july': 7,
         'august': 8,
         'september': 9,
         'october': 10,
         'november': 11,
         'december': 12}
months_reversed = {v: k for k, v in months.items()}
seasons = {'winter': (12, 2),
           'spring': (3, 5),
           'summer': (6, 8),
           'autumn': (9, 11),
           'fall': (9, 11)}

vague_language = ['several', 'some', 'couple']