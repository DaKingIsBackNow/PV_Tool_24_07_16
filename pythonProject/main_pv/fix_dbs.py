import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import os 
import main_pv.solar_functions as solar_functions
import pvlib

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MAX_WIND = 20
MIN_TEMP = -50
MAX_TEMP = 50

def fix_h0():
    """
    Fixes the electricity demand from having 15-min values to 10-min values.
    """
    path = DIR_PATH + "/Databases/Archive/ElDemand.csv"
    el_demand = pd.read_csv(filepath_or_buffer=path, delimiter=";", index_col=0)
    cols = []
    for i in range(24):
        if i < 10:
            str_i = f"0{i}"
        else:
            str_i = f"{i}"
        for j in range(6):
            if i == j == 0:
                continue
            str_j = f"{int(j*10)}"
            if j == 0:
                str_j = "00"
            cols.append(f"{str_i}:{str_j}")
    cols.append("00:00")
    new_el_demando = el_demand[["Day", "Date"]]
    new_el_demando = pd.concat(
        [
            new_el_demando,
            pd.DataFrame(
                [],
                index=new_el_demando.index,
                columns=cols
            )
        ], axis=1
    )
    for i in range(24):
        str_old_hour = f"{i}"
        str_old_next_hour = f"{i+1}"
        if i < 10:
            str_new_hour = f"0{i}"
            if i == 9:
                str_next_hour = "10"
            else:
                str_next_hour = f"0{i + 1}"
        else:
            str_new_hour = f"{i}"
            if i == 23:
                str_next_hour = "00"
                str_old_next_hour = "0"
            else:
                str_next_hour = f"{i + 1}"

        str_time = f"{str_new_hour}:10"
        new_el_demando[str_time] = (el_demand[f"{str_old_hour}:15"] * 2 / 3)

        str_time = f"{str_new_hour}:20"
        new_el_demando[str_time] = (
                el_demand[f"{str_old_hour}:15"] * 1 / 3 + el_demand[f"{str_old_hour}:30"] * 1 / 3)

        str_time = f"{str_new_hour}:30"
        new_el_demando[str_time] = el_demand[f"{str_old_hour}:30"] * 2 / 3

        str_time = f"{str_new_hour}:40"
        new_el_demando[str_time] = el_demand[f"{str_old_hour}:45"] * 2 / 3

        str_time = f"{str_new_hour}:50"
        new_el_demando[str_time] = (
                el_demand[f"{str_old_hour}:45"] * 1 / 3 + el_demand[f"{str_old_next_hour}:00"] * 1 / 3)

        str_time = f"{str_next_hour}:00"
        new_el_demando[str_time] = el_demand[f"{str_old_next_hour}:00"] * 2 / 3
    path = DIR_PATH + "/Databases/Main/Test.csv"
    new_el_demando = new_el_demando.round(1)
    new_el_demando.to_csv(path, sep=";")


def fix_temp():
    path = DIR_PATH + "/Databases/TempStutNew.txt"
    db = pd.read_csv(filepath_or_buffer=path, sep=";")
    # path_temp = dir_path + "/Databases/TempStut.txt"
    # db_temp = pd.read_csv(filepath_or_buffer=path, sep=";")
    db["Datetime"] = pd.to_datetime(db["MESS_DATUM"], format="%Y%m%d%H%M")
    station_id = db["STATIONS_ID"].values[0]
    print(db.head())
    db = db[["Datetime", "TT_10"]]

    counter_error = 0

    for i in range(len(db.index)):
        if db["TT_10"].values[i] == -999:
            db.at[i, "TT_10"] = db["TT_10"].values[i - 1]
            counter_error += 1

    path = DIR_PATH + f"/Databases/Main/Temperature{station_id}.csv"
    # print(f"The global is lower than diffuse {counterDiffbigger} many times")
    # print(f"-999 G {counterG} many times")
    # print(f"-999 D {counterD} many times")
    # dSum = db['DS_10'].sum()
    # gSum = db['GS_10'].sum()
    # print(f"gSum is {gSum}")
    # print(f"dSum is {dSum}")
    # print(f"The average diffus part is {round(dSum/gSum*100, 2)} %")
    # db = db.sort_values(by=['DS_10'], ascending=False)
    print(db.head())
    # print(f"The average diffus part is {db['DS_10'].sum()} %")
    db.to_csv(path, sep=";")  
    # 0.2<x<0.25 1/(1/U_alt + x/d)


def temp_year(year: int):
    path = DIR_PATH + "/Databases/TempStut.txt"
    db = pd.read_csv(filepath_or_buffer=path, sep=";")
    # path_temp = dir_path + "/Databases/TempStut.txt"
    # db_temp = pd.read_csv(filepath_or_buffer=path, sep=";")
    db["Datetime"] = pd.to_datetime(db["MESS_DATUM"], format="%Y%m%d%H%M")
    db = db.loc[db['Datetime'].dt.year == year]
    station_id = db["STATIONS_ID"].values[0]
    print(db.head())
    db = db[["Datetime", "TT_10"]]

    counter_error = 0

    for i in range(len(db.index)):
        if db["TT_10"].values[i] == -999:
            db.at[i, "TT_10"] = db["TT_10"].values[i - 1]
            counter_error += 1

    path = DIR_PATH + f"/Databases/Main/Temperature{station_id}-{year}.csv"
    db = db.reset_index(drop=True)
    # print(f"The global is lower than diffuse {counterDiffbigger} many times")
    # print(f"-999 G {counterG} many times")
    # print(f"-999 D {counterD} many times")
    # dSum = db['DS_10'].sum()
    # gSum = db['GS_10'].sum()
    # print(f"gSum is {gSum}")
    # print(f"dSum is {dSum}")
    # print(f"The average diffus part is {round(dSum/gSum*100, 2)} %")
    # db = db.sort_values(by=['DS_10'], ascending=False)
    print(db.head())
    # print(f"The average diffus part is {db['DS_10'].sum()} %")
    db.to_csv(path, sep=";")
    # 0.2<x<0.25 1/(1/U_alt + x/d)


def fix_solar():
    path = DIR_PATH + "/Databases/SolarStut.txt"
    db = pd.read_csv(filepath_or_buffer=path, sep=";")
    db = db[["STATIONS_ID", "MESS_DATUM", "DS_10", "GS_10"]]
    print(db.head())
    db["Datetime"] = pd.to_datetime(db["MESS_DATUM"], format="%Y%m%d%H%M")
    station_id = db["STATIONS_ID"].values[0]
    db = db[["Datetime", "DS_10", "GS_10"]]

    counterG = 0
    counterD = 0
    counterDiffbigger = 0
    for i in range(len(db.index)):
        if db["GS_10"].values[i] < db["DS_10"].values[i]:
            counterDiffbigger += 1
        if db["GS_10"].values[i] == -999:
            db.at[i, "GS_10"] = db["GS_10"].values[i - 1]
            counterG += 1
        if db["DS_10"].values[i] == -999:
            db.at[i, "DS_10"] = db["GS_10"].values[i] / 2
            counterD += 1

    # d = datetime.strptime("202001010000", "%Y%m%d%H%M")
    # db = db.sort_values(by=['GS_10'], ascending=False)
    # print(db.head())
    path = DIR_PATH + f"/Databases/Main/Solar{station_id}.csv"
    print(f"The global is lower than diffuse {counterDiffbigger} many times")
    print(f"-999 G {counterG} many times")
    print(f"-999 D {counterD} many times")
    dSum = db['DS_10'].sum()
    gSum = db['GS_10'].sum()
    print(f"gSum is {gSum}")
    print(f"dSum is {dSum}")
    print(f"The average diffuse part is {round(dSum/gSum*100, 2)} %")
    # db = db.sort_values(by=['DS_10'], ascending=False)
    print(db.head())
    # db.to_csv(path, sep=";")
    # 0.2<x<0.25 1/(1/U_alt + x/d)


def show_solar_winter():
    path = DIR_PATH + "/Databases/Main/Solar4928.csv"
    df = pd.read_csv(filepath_or_buffer=path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["Datetime"].dt.month < 4) | (df["Datetime"].dt.month >= 10)]
    df['Time'] = df['Datetime'].apply(lambda x: x.time())
    df["Global"] = df["DS_10"] + df["GS_10"]
    df["Global"] = df["Global"]*100/6/10
    df = df[["Time", "Global"]]
    df = df.groupby(["Time"]).mean()

    datetimey = datetime.datetime(hour=0, minute=0, year=2018, month=1, day=1)
    delta = datetime.timedelta(minutes=10)
    x = np.empty(24 * 6, dtype=datetime.datetime)
    y = np.empty(24 * 6)
    for i in range(24 * 6):
        timey = datetimey.time()
        x[i] = datetimey
        datetimey += delta
        y[i] = (df.loc[df.index == timey])["Global"]

    # x = [timmy.strftime("%H:%M") for timmy in x]
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.plot(x, y, 'bo')
    plt.ylabel("Solar radiation in percent of 1000 W/m²")
    plt.xlabel("Time of day")
    plt.title("Winter average total radiation")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    plt.show()
    x = x[0::6]
    x = [item.time().hour for item in x]
    x = pd.Index(x, name="Hour")
    y = y[0::6]
    db_save = pd.DataFrame(index=x, data=y, columns=["GSR"])
    path = DIR_PATH + "/Databases/Seasons/Winter.csv"
    db_save.to_csv(path)


def show_solar_summer():
    path = DIR_PATH + "/Databases/Main/Solar4928.csv"
    df = pd.read_csv(filepath_or_buffer=path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["Datetime"].dt.month >= 4) & (df["Datetime"].dt.month < 10)]
    df['Time'] = df['Datetime'].apply(lambda x: x.time())
    df["Global"] = df["DS_10"] + df["GS_10"]
    df["Global"] = df["Global"]*100/6/10
    df = df[["Time", "Global"]]
    df = df.groupby(["Time"]).mean()

    datetimey = datetime.datetime(hour=0, minute=0, year=2018, month=1, day=1)
    delta = datetime.timedelta(minutes=10)
    x = np.empty(24*6, dtype=datetime.datetime)
    y = np.empty(24*6)
    for i in range(24*6):
        timey = datetimey.time()
        x[i] = datetimey
        datetimey += delta
        y[i] = (df.loc[df.index == timey])["Global"]

    # x = [timmy.strftime("%H:%M") for timmy in x]
    plt.gcf().autofmt_xdate()
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    plt.plot(x, y, 'bo')
    plt.ylabel("Solar radiation in percent of 1000 W/m²")
    plt.xlabel("Time of day")
    plt.title("Summer average total radiation")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    plt.show()
    x = x[0::6]
    x = [item.time().hour for item in x]
    x = pd.Index(x, name="Hour")
    y = y[0::6]
    db_save = pd.DataFrame(index=x, data=y, columns=["GSR"])
    path = DIR_PATH + "/Databases/Seasons/Summer.csv"
    db_save.to_csv(path)


def get_len_float_int_array(value: float|int|np.ndarray|str) -> int:
    """
    Returns the length of an array or single value. If it's not an array, it returns one. 
    Can't return the length of a string because it would return the number of characters and spaces and whatever.
    """
    if type(value) == str:
        return 1
    try:
        return len(value)
    except TypeError:  # if it's an int or float
        return 1


def remove_abnormalities(df: pd.DataFrame, min_valid_val: float|int|np.ndarray, max_valid_val: float|int|np.ndarray, col_string:str|np.ndarray, default_to_min: bool|np.ndarray = True) -> pd.DataFrame:
    """
    Gives back a fixed kingdom, free from the cage of corruption. Each monarch is evaluated and their treasuries are dutyfully controlled, supplying much needed transparency to the kingdom.
    (Gives you back a database with fixed columns to stay within a valid range. It takes the previous valid value and applies it to abnormal cells within the dataframe)
    Args:
            :param df: the kingdom of data (The Dataframe).
            :param min_valid_val: A list of the lowest possible value for each monarch's treasure trove in kg of gold (array of min values for the "normal" range in order).
            :param max_valid_val: A list of the highest possible value for each monarch's treasure trove in kg of gold (array of max values for the "normal" range in order).
            :param col_string: the names of the monarchs (names of the columns that are wished to be fixed)
            :param default_to_min: Do the monarchs default to bankrupcy or get bailed out by the state? (default to the minimum value if the first instance of the database is abnormal.)

        Returns:
            pd DataFrame: A united kingdom (not that one) that is ready to take on the modern world and pay their share of taxes. (the fixed dataframe) 
    """

    """
    Validation
    """
    len_min = get_len_float_int_array(min_valid_val)
    len_max = get_len_float_int_array(max_valid_val)
    len_col = get_len_float_int_array(col_string)
    if len_min == len_max == len_col:
        if get_len_float_int_array(default_to_min) == 1 and get_len_float_int_array(default_to_min) != get_len_float_int_array(col_string):
            default_to_min = np.tile(default_to_min, get_len_float_int_array(col_string))
        else:
            if get_len_float_int_array(default_to_min) != get_len_float_int_array(col_string):
                raise ValueError("Your stuff don't match dawg. Check it. (array of default_to_min isn't same length as col_string)")
    else:
        raise ValueError("Your stuff don't match dawg. Check it. (arrays not the same length as each other)")
    
    len_data = get_len_float_int_array(col_string)
    if len_data == 1:
        """
        Case for one column
        """
        for i in range(len(df.index)):
            if df[col_string].values[i] < min_valid_val or df[col_string].values[i] > max_valid_val:
                if i == 0:
                    df.at[df.index[i], col_string] = int(default_to_min)*min_valid_val - (int(default_to_min)-1)*max_valid_val
                else:
                    df.at[df.index[i], col_string] = df.at[df.index[i-1], col_string]
        return df
    """
    case for multiple columns in need of changing. No need for else because return in previous thingy.
    """
    for i in range(len(df.index)):
        for j in range(len_data):
            if df[col_string[j]].values[i] < min_valid_val[j] or df[col_string[j]].values[i] > max_valid_val[j]:
                if i == 0:
                    df.at[df.index[i], col_string[j]] = int(default_to_min[j])*min_valid_val[j] - (int(default_to_min[j])-1)*max_valid_val[j]
                else:
                    df.at[df.index[i], col_string[j]] = df.at[df.index[i-1], col_string[j]]
    return df


def get_wind_year(year: int) -> np.ndarray:
    path_wind = DIR_PATH + "/Databases/WindStut.txt"
    db_wind = pd.read_csv(filepath_or_buffer=path_wind, sep=";")
    db_wind["Datetime"] = pd.to_datetime(db_wind["MESS_DATUM"], format="%Y%m%d%H%M")
    db_wind = db_wind.loc[db_wind['Datetime'].dt.year == year]
    
    db_wind = remove_abnormalities(db_wind, 0, MAX_WIND, "FF_10")

    vals = db_wind["FF_10"].to_numpy(dtype=float)

    return vals


def get_temp_year(year: int) -> np.ndarray:
    path_temp = DIR_PATH + "/Databases/TempStut.txt"
    db_temp = pd.read_csv(filepath_or_buffer=path_temp, sep=";")
    db_temp["Datetime"] = pd.to_datetime(db_temp["MESS_DATUM"], format="%Y%m%d%H%M")
    db_temp = db_temp.loc[db_temp['Datetime'].dt.year == year]

    db_temp = remove_abnormalities(db_temp, MIN_TEMP, MAX_TEMP, "TT_10")

    vals = db_temp["TT_10"].to_numpy(dtype=float)

    return vals



def solar_year(year: int):
    path_sol = DIR_PATH + "/Databases/SolarStut.txt"
    db_sol = pd.read_csv(filepath_or_buffer=path_sol, sep=";")

    db_sol = db_sol[["STATIONS_ID", "MESS_DATUM", "DS_10", "GS_10"]]
    print(db_sol.head())
    db_sol["Datetime"] = pd.to_datetime(db_sol["MESS_DATUM"], format="%Y%m%d%H%M")
    station_id = db_sol["STATIONS_ID"].values[0]
    db_sol = db_sol.loc[db_sol['Datetime'].dt.year == year]
    db_sol = db_sol[["Datetime", "DS_10", "GS_10"]]
    db_sol = db_sol.reset_index(drop=True)
 
    counterG = 0
    counterD = 0
    counterDiffbigger = 0

    for i in range(len(db_sol.index)):
        if db_sol["GS_10"].values[i] < db_sol["DS_10"].values[i]:
            counterDiffbigger += 1
        if db_sol["GS_10"].values[i] == -999:
            db_sol.at[i, "GS_10"] = db_sol["GS_10"].values[i - 1]
            counterG += 1
        if db_sol["DS_10"].values[i] == -999:
            db_sol.at[i, "DS_10"] = db_sol["GS_10"].values[i] / 2
            counterD += 1
    
    db_sol["dhi"] = (db_sol["DS_10"]*10000/3600*6).round(2)  # converting J/cm²/10min to average W/m² aka J/s/m²
    db_sol["ghi"] = (db_sol["GS_10"]*10000/3600*6).round(2)  # converting J/cm²/10min to average W/m² aka J/s/m²
    days_array = db_sol["Datetime"].dt.day_of_year.to_numpy(dtype=int)
    times_array = ((db_sol["Datetime"].dt.hour + db_sol["Datetime"].dt.minute/60)/24).to_numpy(dtype=float)
    ghi_array = db_sol["ghi"].to_numpy(dtype=float)
    dhi_array = db_sol["dhi"].to_numpy(dtype=float)
    zenith = solar_functions.get_zeniths(days_array, times_array)

    db_sol["dni"] = np.round(solar_functions.get_dni(days_array, times_array,ghi_array,dhi_array), 2)
    # db_sol["dni_pvlib"] = pvlib.irradiance.complete_irradiance(ghi=ghi_array, dhi=dhi_array, solar_zenith=zenith)["dni"]
    # db_sol = db_sol[["Datetime", "ghi", "dhi", "dni", "dni_pvlib"]]
    db_sol = db_sol[["Datetime", "ghi", "dhi", "dni"]]
    
    db_sol = db_sol.set_index("Datetime", drop=True)
    db_sol["wind_speed"] = get_wind_year(year)
    db_sol["temp_air"] = get_temp_year(year)
    path_sol = DIR_PATH + f"/Databases/Main/Solar{station_id}-{year}.csv"
    print(db_sol.head())
    db_sol.to_csv(path_sol, sep=";")


# show_solar_summer()
# show_solar_winter()
# temp_year(2023)
solar_year(2023)
print(7)
# fix_solar()
# fix_temp()
