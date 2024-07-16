import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'
heat_temp_limit = 15  # Above this point the building is not being heated.


def analyze_heat_pump_performance():
    """
    Saves some analysis to an Excel file.
    """
    year_start = 1980
    year_ender = 2023
    bivs = [-20, -5, -2, 2, 5, 10]  # Lowest temperature, where the heat pump still works
    ww_per = 5  # Percentage of warm water heat demand out of total
    global heat_temp_limit
    db = pd.read_excel("H:/Dokumentos/Alt/Years.xlsx", sheet_name="Since 1980")
    save_database = pd.DataFrame({"Year": [], "C": [], "B": [], "ww": [], "jaz": []})

    while year_start < year_ender and not fetch_yearly_records(db, year_start).empty:
        for biv in bivs:
            df = fetch_yearly_records(db, year_start)
            df = df.loc[df["T"] < heat_temp_limit]
            # sumpro = 0
            # summer = 0
            flow_temp_classes = [35, 45, 55, 60]
            warm_water_efficiency = sum([df["cop60"][i] * df["Hours"][i] for i in range(len(df.index))]) / df["Hours"].sum()
            for flow_class in flow_temp_classes:
                df = df.loc[df["T"] > biv]
                print(f"The amount of hours the heat pump will work is equal to")
                df.index = range(len(df.index))
                class_string = f"cop{flow_class}"
                sumpro = sum([df[class_string][i] * df["Hours"][i] for i in range(len(df.index))])
                summer = df["Hours"].sum()
                # for i in range(len(tempie.index)):
                #     sumpro += tempie[class_string][start+i]*tempie["Hours"][start+i]
                #     summer += tempie["Hours"][start+i]
                jaz = sumpro / summer
                jaz = ww_per / 100 * warm_water_efficiency + (1 - ww_per / 100) * jaz
                row = {"Year": [year_start], "C": [flow_class], "B": [biv], "ww": [ww_per], "jaz": jaz}
                save_database = pd.concat([save_database, pd.DataFrame(row)])
            year_start += 1

    save_database.columns = ["Year", "Flow Temp (°C)", "Bivalent Point (°C)",
                            "Warm water share (%)", "Yearly Efficiency"]
    # with pd.ExcelWriter("C:/Users/Daniel/Downloads/Years.xlsx", mode='a', if_sheet_exists='replace') as writer:
    #     save_database.to_excel(writer, sheet_name="JAZ")


def fetch_yearly_records(db, year):
    df = db[db["Year"] == year]
    df.index = range(len(df.index))
    return df


def calculate_jaz_main(year_start, year_ender, vor, bivv, ww, ww_temp):
    """
    Main efficiency calculating function
    :param year_start: First year of weather data
    :param year_ender: Last year of weather data
    :param vor: flow temperature in °C
    :param bivv: bivalence temperature in °C
    :param ww: Share of warm water heat production of total heat production
    :param ww_temp: Temperature of the water in the warm water tank
    :return: the yearly efficiency of the heat pump with the given variables.
    """
    classes = np.array([35, 45, 55, 60])
    if vor < 35 or vor > 60:
        return "I'mma be honest with you dog, I don't know how to do that."
    elif vor in classes:
        return calculate_jaz_helper(year_start, year_ender, vor, bivv, ww, ww_temp)
    else:
        classy = [classes[classes < vor].max(), classes[classes > vor].min()]
        val_low = calculate_jaz_helper(year_start, year_ender, classy[0], bivv, ww, ww_temp)
        val_high = calculate_jaz_helper(year_start, year_ender, classy[0], bivv, ww, ww_temp)
        dif = classy[1] - classy[0]
        return val_low + (val_high-val_low)/dif*(vor-classy[0])


def calculate_jaz_helper(year_start, year_ender, vor, bivv, ww, ww_temp):
    """
        Helper function for 'calculate_jaz_main'
        :param year_start: First year of weather data
        :param year_ender: Last year of weather data
        :param vor: flow temperature in °C
        :param bivv: bivalence temperature in °C
        :param ww: Share of warm water heat production of total heat production
        :param ww_temp: Temperature of the water in the warm water tank
        :return: the yearly efficiency of the heat pump with the given variables.
    """
    biv = [bivv]  # Lowest temperature, where the heat pump still works
    ww_per = ww  # Percentage of warm water heat demand out of total
    db = pd.read_excel("H:/Dokumentos/Alt/Years.xlsx", sheet_name="Since 1980")
    jazz = []
    classy = [vor]

    while year_start < year_ender and not fetch_yearly_records(db, year_start).empty:
        for vallie in biv:
            df = fetch_yearly_records(db, year_start)
            j = 0
            if ww_per == 0:
                jaz_ww = 0
            else:
                jaz_ww = calculate_jaz_main(year_start, year_ender, ww_temp, -30, 0, 0)
            for value in classy:
                df = df.loc[df["T"] > vallie]
                df.index = range(len(df.index))
                classical = f"cop{value}"
                energy = sum([df[classical][i] * df["Hours"][i] for i in range(len(df.index))])
                hours = df["Hours"].sum()
                jazz.append(energy / hours)
                jazz[-1] = ww_per / 100 * jaz_ww + (1 - ww_per / 100) * jazz[j]
                j += 1
            year_start += 1

    return sum(jazz) / len(jazz)


def calculate_jaz_2021_helper(year_start, year_ender, vor, bivv, ww, ww_temp):
    """
    Helper efficiency calculating function for the year 2021
    :param year_start: First year of weather data
    :param year_ender: Last year of weather data
    :param vor: flow temperature in °C
    :param bivv: bivalence temperature in °C
    :param ww: Share of warm water heat production of total heat production
    :param ww_temp: Temperature of the water in the warm water tank
    :return: the yearly efficiency of the heat pump with the given variables.
    """
    biv = [bivv]  # Lowest temperature, where the heat pump still works
    ww_per = ww  # Percentage of warm water heat demand out of total
    db = pd.read_excel("H:/Dokumentos/Alt/Years.xlsx", sheet_name="2021")
    jaz = []
    classy = [vor]

    while year_start < year_ender and not fetch_yearly_records(db, year_start).empty:
        for vallie in biv:
            df = fetch_yearly_records(db, year_start)
            j = 0
            if ww_per == 0:
                jaz_ww = 0
            else:
                jaz_ww = calculate_jaz_main(year_start, year_ender, ww_temp, -30, 0, 0)
            for value in classy:
                df = df.loc[df["T"] > vallie]
                df.index = range(len(df.index))
                classical = f"cop{value}"
                sumpro = sum([df[classical][i] * df["Hours"][i] for i in range(len(df.index))])
                summer = df["Hours"].sum()
                jaz.append(sumpro / summer)
                jaz[-1] = ww_per / 100 * jaz_ww + (1 - ww_per / 100) * jaz[j]
                j += 1
            year_start += 1

    return sum(jaz) / len(jaz)


def calculate_jaz_2021_main(year_start, year_ender, vor, bivv, ww, ww_temp):
    """
    Main efficiency calculating function for 2021
    :param year_start: First year of weather data
    :param year_ender: Last year of weather data
    :param vor: flow temperature in °C
    :param bivv: bivalence temperature in °C
    :param ww: Share of warm water heat production of total heat production
    :param ww_temp: Temperature of the water in the warm water tank
    :return: the yearly efficiency of the heat pump with the given variables.
    """
    classes = np.array([35, 45, 55, 60])
    if vor < 35 or vor > 60:
        return "I'mma be honest with you dog, I don't know how to do that."
    elif vor in classes:
        return calculate_jaz_2021_helper(year_start, year_ender, vor, bivv, ww, ww_temp)
    else:
        classy = [classes[classes < vor].max(), classes[classes > vor].min()]
        val_low = calculate_jaz_2021_helper(year_start, year_ender, classy[0], bivv, ww, ww_temp)
        val_high = calculate_jaz_2021_helper(year_start, year_ender, classy[0], bivv, ww, ww_temp)
        dif = classy[1] - classy[0]
        return val_low + (val_high-val_low)/dif*(vor-classy[0])


def initialize_analysis_parameters():
    """
    Initializes the default parameters for the heat pump.
    :return: a Tuple with the variables needed to calculalte the heat pump.
    """
    year_begin = 2010
    year_end = 2022
    flow_temp = 55
    biv_point = 5
    ww_tank_temp = 60  # °C

    ww_dem = 23.83  # kWh/m²a
    ht_dem = 167.5  # kWh/m²a
    ww_wp = 95  # % of warm water produced by the waterpump
    ht_wp = 95  # % of heat demand produced by the waterpump
    ww_wp, hz_wp = [ww_wp / 100, ht_wp / 100]

    ww_share = (ww_wp * ww_dem / (ht_dem * hz_wp + ww_dem * ww_wp)) * 100  # percent of
    # warm water production from the total work of the pump
    # ww_share = 0
    return year_begin, year_end, flow_temp, biv_point, ww_share, ww_tank_temp


# summary()

var = initialize_analysis_parameters()
jaz = calculate_jaz_main(var[0], var[1], var[2], var[3], var[4], var[5])

print(round(jaz, 2))

Ht = 0.7

# Classy = [35, 45, 55, 60]
# y = []
# biv = 2
# x = Classy
# for year_begin in np.arange(2010, 2013, 1):
#     y.append([get_jaz(year_begin, year_begin+1, cl, biv, ww_anteil, ww_tank_temp) for cl in Classy])
# plt.plot(x, y[0], 'ro', x, y[1], 'bo', x, y[2], 'yo')
# plt.ylabel("JAZ of the heat pump")
# plt.xlabel("Desired water temperature")
# plt.legend(np.arange(2018, 2021, 1))
# plt.show()

# cl = 35
# x = np.arange(1990, 2021, 1)
# for year_begin in np.arange(1990, 2021, 1):
#     y.append(get_jaz(year_begin, year_begin+1, cl, biv, ww_anteil, ww_tank_temp))
# plt.plot(x, y, 'ro')
# plt.ylabel("JAZ of the heat pump")
# plt.xlabel("Year")
# plt.plot(x, y, 'ro')

# y = []
# cl = 35
# x = np.arange(-2, 3, 1)
# y.append([get_jaz(2021, 2022, cl, biv, ww_anteil, ww_tank_temp) for biv in x])
# y.append([get_jaz_2021(2021, 2022, cl, biv, ww_anteil, ww_tank_temp) for biv in x])
# plt.plot(x, y[0], 'ro', x, y[1], 'bo')
# plt.ylabel("JAZ of the heat pump")
# plt.xlabel("Bivalent point")
# plt.legend(['0.5', '0.1'])
# plt.show()

