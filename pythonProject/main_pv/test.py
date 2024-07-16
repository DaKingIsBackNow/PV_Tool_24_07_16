import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import datetime
import time

start_time = time.time()

DEBUGGING = True
YEAR_10_MIN_DATALENGTH = 52704
Week_10_MIN_DATALENGTH = 52704
WEEK_INDNAMES = ["Weekday", "Time"]
YEAR_INDNAMES = ["Date", "Time"]
ELDEMAND_STRING = "ElDemand"

if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


def calculate_max_subsidy(residential_units: int):
    """
    Returns the max amount of money eligible for subsidies by the amount of living units.
    The calculation is 30.000 € for the first unit, 15.000 € for the second to sixth including,
    and 8.000 € for the rest.
    :param: residential_units: The amount of residential units.
    :return: The max amount eligible for subsidy in €.
    """
    first_subsidy_amount = 30000
    second_subsidy_amount = 15000
    third_subsidy_amount = 8000
    first_threshold = 1
    second_threshold = 6

    try:
        residential_units = int(residential_units)
    except ValueError:
        raise ValueError("You a bitch!")

    if not isinstance(residential_units, int) or residential_units <= 0:
        return 0

    if residential_units <= second_threshold:
        subsidy = (first_subsidy_amount * first_threshold +
                   second_subsidy_amount * (residential_units - first_threshold))
    else:
        subsidy = (first_subsidy_amount * first_threshold +
                   second_subsidy_amount * (second_threshold - first_threshold) +
                   third_subsidy_amount * (residential_units - second_threshold))
    return subsidy


# el_demand = pd.read_csv("Databases/Archive/ElDemand.csv", delimiter=";", index_col=0)
# print(el_demand.iloc[:, 2].head())

def call_max_sub():
    living_thingies = np.arange(0, 13, 1)
    subbies = [calculate_max_subsidy(liv) for liv in living_thingies]
    plt.plot(living_thingies, subbies, 'b--')
    plt.show()
    print({key: value for key, value in zip(living_thingies, subbies)})


def db_thingies():
    path = "Databases/Archive/Solar.csv"

    db = pd.read_csv(filepath_or_buffer=path, sep=";")
    db["Datetime"] = pd.to_datetime(db["Datetime"])
    print(db["Datetime"].dt.dayofyear[0])


def testnumpie():
    vecky1 = np.array([10, 8, 2])
    vecky2 = np.array([-10, 3, 9])
    return np.matmul(vecky1, vecky2)


def test_forloop():
    lister = [12, 100, 92, 5, 1034]

    for ind, val in enumerate(lister):
        if ind == len(lister)-1:
            next_val = lister[0]
        else:
            next_val = lister[ind+1]
        print(f"index {ind}, Current val: {val}, next val: {next_val}")


def test_date_time_time():
    weeday_now = 2
    weeday_next = 1
    time_now = datetime.time(hour=8, minute=30)
    decimal_time_now = (time_now.hour + time_now.minute/60)/24
    time_next = datetime.time(hour=10, minute=30)
    decimal_time_next = (time_next.hour + time_next.minute/60)/24

    time_dif = weeday_next - weeday_now + decimal_time_next - decimal_time_now
    if time_dif < 0:
        time_dif += 7

    print(datetime.timedelta(hours=31).seconds/3600)

    print(f"The time difference is {round(time_dif*24, 0)} hours")


def validate_weekday(weekday: int):
    """
    Validates the weekday.
    """
    if not 0 <= weekday <= 6:
        raise ValueError("A week goes from 0 to 6 here. Either you tried 7 or "
                         "you're a rebel that works with weeks longer than 7 days.")


def validate_time_away(time_away: datetime.timedelta):
    """
        Validates the time away on a weekly trip.
    """
    if not time_away.days < 7:
        raise ValueError("Going on a weekly trip that takes longer than a week isn't something we support. "
                         "Individuals with time manipulation abilities might need a different service.")


def get_time_back(weekday: int, time_away: datetime.timedelta, departure: datetime.time):
    """
    returns the time back after a trip with the car and the weekday in case the trip flows over one or more
    midnights of returning for a weekly event.
        :param departure: departure time i.e. 20:30:00
        :param time_away: how long until the car is back in the building.
        :param weekday: 0 = Monday 6 = Sunday
        :return: A tuple with time back in the first position and weekday in the second.
        The weekday returned follows the same convention of monday being 0.
    """
    validate_weekday(weekday)
    validate_time_away(time_away)
    # no point validating the departure time, since any time is fine, and it is already in the format datetime.time.
    # I'm not one to judge people for a trip at 3 AM. You do you boo.

    vir_time_depart = datetime.datetime(year=2024, month=1, day=weekday+1, hour=departure.hour, minute=departure.minute)
    # let's pretend we're in the beginning of 2024 to work with datetime objects. The first of January was
    # even a monday, which works great with our weekday convention.
    vir_time_back = vir_time_depart + time_away
    return vir_time_back.time(), vir_time_back.weekday()  # since we synchronized the weekday with 2024 we can just
    # look at the weekday after coming back from the trip, and it would be the same for every week, except for some
    # Daylight saving time shenanigans.


def get_weekly_schedule():
    """
    Creates a weekly schedule of electric chargin and returns the dataframe. Using 2024 because it started on a monday.
    The year doesn't get saved later and plays no part in the returned database.
    """
    datey_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    deltie = datetime.timedelta(minutes=10)
    data = []
    indies = []
    while datey_time.day < 8:
        # Here comes the logic in the real function. Here we just have the if statement for testing.
        if datey_time.weekday() == 0:
            data.append(2.3)
        else:
            data.append(6.5)
        indies.append((datey_time.weekday(), datey_time.time()))
        datey_time += deltie
    index = pd.MultiIndex.from_tuples(indies, names=["Weekday", "Time"])
    db_week = pd.DataFrame(columns=["ElDemand"], index=index, data=data)
    return db_week


def test_get_weekly_schedule():
    """
    Creates a weekly schedule of electric chargin and returns the dataframe. Using 2024 because it started on a monday.
    The year doesn't get saved later and plays no part in the returned database.
    """
    values = [12, 102, 3, 15]
    indexes = [(0, datetime.time(hour=0, minute=10)),
               (0, datetime.time(hour=0, minute=40)),
               (2, datetime.time(hour=0, minute=30)),
               (6, datetime.time(hour=4, minute=10))]
    indexi = pd.MultiIndex.from_tuples(indexes, names=WEEK_INDNAMES)
    db_real_charge = pd.DataFrame(data=values, index=indexi, columns=[ELDEMAND_STRING])
    
    datey_time = np.datetime64(f"{2024}-01-01T00:00")
    deltie = np.timedelta64(10, "m")

    date_start = datey_time
    date_stop = date_start + np.timedelta64(7, "D")
    indies = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))
    data = np.empty(dtype=float, shape=len(indies))

    indies = [[d.day-1, d.time()] for d in indies]

    for i in range(len(indies)):
        # Here comes the logic in the real function. Here we just have the if statement for testing.
        data[i] = 0

    index = pd.MultiIndex.from_tuples(indies, names=WEEK_INDNAMES)
    db_week = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=data)
    db_week.update(db_real_charge)
    return db_week


def get_yearly_schedule(year: int):
    """
    Creates a yearly schedule by the weekly schedule. Returns a dataframe. Daytime saving time was a mistake. We do
    not believe in it.
    """
    """
    Initializing
    """
    db_week = test_get_weekly_schedule()
    print(db_week.head())
    datey_time = np.datetime64(f"{year}-01-01T00:00")
    weekday_start = pd.to_datetime(datey_time).weekday()
    deltie = np.timedelta64(10, "m")

    date_start = datey_time
    date_stop = np.datetime64(f"{year+1}-01-01T00:00")
    indies = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))
    indies = [[d.date(), d.time()] for d in indies]
    datalength = len(indies)
    data = np.empty(dtype=float, shape=datalength)

    db_column = db_week.loc[:, ELDEMAND_STRING].to_numpy()
    len_db_column = len(db_column)
    """
    Week 1
    """
    start_ind_weekly = int(len_db_column/7*weekday_start)  # If we start on monday, this is zero. If we start on tuesday this is 24*6=144 if the intervals are ten minutes. 
    end_index_first_week = len_db_column - start_ind_weekly  # This is the index for the end of week 1 of the year. It should be the length of a week minus the steps we skip.
    rest_datalength = datalength - end_index_first_week

    repeat_count = rest_datalength // len_db_column 
    remainder = rest_datalength % len_db_column

    data[:end_index_first_week] = db_column[start_ind_weekly:]
    data[end_index_first_week:end_index_first_week + repeat_count * len_db_column] = np.tile(db_column, repeat_count)
    data[end_index_first_week + repeat_count * len_db_column:] = db_column[:remainder] 

    index = pd.MultiIndex.from_tuples(indies, names=YEAR_INDNAMES)
    db_year = pd.DataFrame(index=index, columns=[ELDEMAND_STRING], data=data)
    return db_year.head()


def get_winter_avg():
    path = "Databases/Seasons/Winter.csv"
    df = pd.read_csv(filepath_or_buffer=path)
    avg = df["Solarstrahl"].values
    return avg


# print(get_winter_avg())


logging.debug(get_yearly_schedule(2020))
# logging.debug(get_winter_avg())
end_time = time.time()
logging.debug(end_time-start_time)
