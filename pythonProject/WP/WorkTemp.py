import pandas as pd
# import numpy as np
DB = pd.read_csv("H:/Dokumentos/Alt/Temperature.csv", decimal=",", sep=";")
WP = pd.read_excel("H:/Dokumentos/Alt/WP Cop.xlsx")
heat_temp_limit = 15  # Above this point the building is not being heated.


def get_heat_pump_heat_output(vt, ta):   # Vorlauftemperatur, Temperatur Außen
    """

    :param vt: heat circuit flow temperature in °C
    :param ta: heat circuit return temperature in °C
    :return: The heating power output in kW at this moment.
    """
    wp_loc = WP.copy()
    wp_loc.columns = ["V", 'T', 'P']
    wp_loc.loc[:, "T"] = wp_loc["T"].apply(lambda x: x.replace(',', '.')).apply(float)
    wp_loc.loc[:, "P"] = wp_loc["P"].apply(lambda x: x.replace(',', '.')).apply(float)
    liston = wp_loc[wp_loc["V"] == vt]
    liston.index = range(len(liston.index))
    minnie = len(liston.index)-1
    maxie = 0
    max_temp = liston["T"][maxie]
    min_temp = liston["T"][minnie]
    if len(liston) == 0:
        return 0
    if ta > max_temp:
        return liston["P"][maxie]
    if ta < min_temp:
        return liston["P"][minnie]
    lowerneighbour_ind = liston[liston["T"] < ta]["T"].idxmax()
    upperneighbour_ind = liston[liston["T"] > ta]["T"].idxmin()
    low_p = liston["P"][lowerneighbour_ind]
    up_p = liston["P"][upperneighbour_ind]
    low_t = liston["T"][lowerneighbour_ind]
    up_t = liston["T"][upperneighbour_ind]
    return low_p + (up_p-low_p)/(up_t-low_t)*(ta-low_t)


def get_heat_pump_el_demand(vt, ta):
    """

        :param vt: heat circuit flow temperature in °C
        :param ta: heat circuit return temperature in °C
        :return: The electric power needed in kW at this moment.
        """
    wp_loc = WP.copy()
    wp_loc.columns = ["V", 'T', 'P']
    wp_loc.loc[:, "T"] = wp_loc["T"].apply(lambda x: x.replace(',', '.')).apply(float)
    # wp_loc["P"] = wp_loc["P"].apply(lambda x: x.replace(',', '.')).apply(float)
    liston = wp_loc[wp_loc["V"] == vt]
    liston.index = range(len(liston.index))
    minnie = len(liston.index)-1
    maxie = 0
    max_temp = liston["T"][maxie]
    min_temp = liston["T"][minnie]
    if len(liston) == 0:
        return 0
    if ta > max_temp:
        return liston["P"][maxie]
    if ta < min_temp:
        return liston["P"][minnie]
    lowerneighbour_ind = liston[liston["T"] < ta]["T"].idxmax()
    upperneighbour_ind = liston[liston["T"] > ta]["T"].idxmin()
    low_p = liston["P"][lowerneighbour_ind]
    up_p = liston["P"][upperneighbour_ind]
    low_t = liston["T"][lowerneighbour_ind]
    up_t = liston["T"][upperneighbour_ind]
    return low_p + (up_p - low_p) / (up_t - low_t) * (ta - low_t)


def get_efficiency(vt, ta):
    """

    :param vt: heat circuit flow temperature in °C
    :param ta: heat circuit return temperature in °C
    :return: The COP value of the heat pump at this moment.
    """
    wp_loc = WP.copy()
    wp_loc.columns = ["V", 'T', 'P']
    # wp_loc["T"] = wp_loc["T"].apply(lambda x: x.replace(',', '.')).apply(float)
    # wp_loc["P"] = wp_loc["P"].apply(str)
    # wp_loc["P"] = wp_loc["P"].apply(lambda x: x.replace(',', '.')).apply(float)
    wp_loc["T"] = wp_loc["T"].apply(float)
    wp_loc["P"] = wp_loc["P"].apply(float)
    liston = wp_loc[wp_loc["V"] == vt]
    liston.index = range(len(liston.index))

    if len(liston) == 0:
        return 0
    if ta > liston["T"][0]:
        return liston["P"][0]
    if ta < liston["T"][len(liston.index) - 1]:
        return liston["P"][len(liston.index) - 1]
    if not liston[liston["T"] == ta]["T"].empty:
        return liston[liston["T"] == ta]["P"].sum()
    lowerneighbour_ind = liston[liston["T"] < ta]["T"].idxmax()
    upperneighbour_ind = lowerneighbour_ind-1
    low_p = liston["P"][lowerneighbour_ind]
    up_p = liston["P"][upperneighbour_ind]
    low_t = liston["T"][lowerneighbour_ind]
    up_t = liston["T"][upperneighbour_ind]
    return low_p + (up_p - low_p) / (up_t - low_t) * (ta - low_t)


def efficiency_across_years():
    """

    :return:
    """
    year_start = 1980
    year_end = 2023
    global DB
    db = DB.copy()
    db.loc[:, "T"] = db["T"].apply(float)
    save_database = get_rounded_halves(db, year_start)
    indo = len(save_database.index)
    cur_year = year_start + 1
    while cur_year < year_end:
        tempie = get_temp_freq(db, cur_year, indo)
        save_database = pd.concat([save_database, tempie])
        indo = len(save_database.index)
        cur_year += 1
    indy = list(save_database["T"])
    save_database["cop35"] = [get_efficiency(35, i) for i in indy]
    save_database["cop45"] = [get_efficiency(45, i) for i in indy]
    save_database["cop55"] = [get_efficiency(55, i) for i in indy]
    save_database["cop60"] = [get_efficiency(60, i) for i in indy]
    save_database = save_database[["T", "Year", "Hours", "cop35", "cop45", "cop55", "cop60"]]
    save_database = save_database.set_index("T", "Year")
    # nine.columns = ["Hours", "cop35", "cop45", "cop55", "cop60"]
    with pd.ExcelWriter("H:/Dokumentos/Alt/Years.xlsx", mode='a', if_sheet_exists='replace') as writer:
        save_database.to_excel(writer, sheet_name="Since 1980 Test")


def get_temp_freq(db, year, indo):
    """

    :param db: the database
    :param year: the year that wants to be returned
    :param indo:
    :return:
    """
    global heat_temp_limit
    eighty = db[db["Year"] == year]
    eighty.loc[:, "T"] = eighty["T"].apply(float)
    eighty = eighty[eighty["T"] < heat_temp_limit]
    tempie = (eighty.groupby(["T"]).count())
    tempie["Year"] = year
    tempie = tempie[["Year", "Month"]]
    tempie.columns = ["Year", "Hours"]
    tempie["Index"] = range(indo, indo+len(tempie.index))
    tempie["T"] = tempie.index
    tempie = tempie.set_index("Index")
    print(tempie.head())
    return tempie


tester = get_temp_freq(DB, 2000, 0)


def get_rounded_halves(db, year):
    """

    :param db: the database
    :param year: the year that wants to be returned
    :return:
    """
    db["T"] = round(db["T"] * 2) / 2
    eighty = db[db["Year"] == year]
    nine = eighty.groupby(["T"]).count()
    nine = nine[["Year", "Month"]]
    nine.columns = ["Year", "Hours"]
    nine["Index"] = range(len(nine.index))
    nine["T"] = nine.index
    nine["Year"] = year
    nine = nine.set_index("Index")
    return nine


efficiency_across_years()
# vtt = 35
# taa = -15
# print(cop(vtt, taa))
# print(pth(vtt, taa)/p_el(vtt, taa))
