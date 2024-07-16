from main_pv.solar_functions import get_percent_onmodule, get__array_rad_onmodule
from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location as PVLocation
from pvlib.pvsystem import PVSystem
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from main_pv.standard_variables import *

# datey = datetime(year=2021, month=6, day=21, hour=13)
# print(get_percent_onmodule(datey, module_angle=35, module_azimuth= 180, timezone=2, total_rad=1000, diff_rad=300))


def get_azimuth_array(azi_interval : float|int, angle_steps: int, data_length: int) -> np.ndarray:
    """
    :param azi_interval: the azimuth step interval for the database.
    :param angle_steps: how many different angles are evaluated per azimuth.
    :param data_length: the total length of the database
    :return: an array of the azimuth values for a future index in a dataframe.
    """
    azimuth_temp = np.empty(shape=data_length, dtype=float)  # empty array because probably more effiecient.
    # azimuth_temp[0:48] = np.repeat(180, 48)  # the first 48 are when the module is flat. Since that is only calculated once for south, 
    # it is not repeated and initialized here. It's the same for any other azimuth like west or east because flat is flat.
    # Should be anyway.
    # azi_help = np.arange(start=180, stop=-180, step=-azi_interval)  # creating an array of all the values of azimuths that will be evaluated.
    azi_help = np.arange(start=-180, stop=180+azi_interval, step=azi_interval)  # creating an array of all the values of azimuths that will be evaluated.
    # azimuth_temp[48:] = np.repeat(azi_help, angle_steps*48)  # adds every azimuth angle_steps*48 times because the summer will have 0-23 hours and so does the winter.
    azimuth_temp = np.repeat(azi_help, angle_steps*2)

    return azimuth_temp


def get_angle_array(angle_interval: float|int, azi_steps: int, data_length: int) -> np.ndarray:
    """
    :param angle_interval: the angle step interval for the database.
    :param azi_steps: how many different azimuths are evaluated per angle.
    :param data_length: the total length of the database
    :return: an array of the angle values for a future index in a dataframe.
    """
    angle_temp = np.empty(shape=data_length, dtype=float)  # empty array because probably more effiecient.
    # angle_temp[0:48] = np.repeat(0, 48)  # the first 48 are when the module is flat. Since that is only calculated once for south, 
    # it is not repeated and initialized here. It's the same for any other azimuth like west or east because flat is flat.
    # Should be anyway.
    # azi_help = np.arange(start=angle_interval, stop=90+angle_interval, step=angle_interval)  # creating an array of all the values of azimuths that will be evaluated.
    angle_help = np.arange(start=0, stop=90+angle_interval, step=angle_interval)  # creating an array of all the values of azimuths that will be evaluated.
    # azi_help = np.repeat(azi_help, 48)
    angle_help = np.repeat(angle_help, 2)
    angle_temp = np.tile(angle_help, azi_steps)
    # angle_temp[48:] = np.tile(angle_help, azi_steps)  # adds every azimuth angle_steps*48 times because the summer will have 0-23 hours and so does the winter.
    # angle_temp = np.tile(angle_help, azi_steps)  # adds every azimuth angle_steps*48 times because the summer will have 0-23 hours and so does the winter.

    return angle_temp


def get_seasons(data_length: int) -> np.ndarray:
    """
    :param data_length: the total length of the database
    :return: an array of the seasons for a future index in a dataframe.
    """
    # summer = np.repeat(SUMMER_STRING, 24)
    # winter = np.repeat(WINTER_STRING, 24)
    # seasons = np.concatenate((summer, winter))  # creating a 48 item long array of 24 "Summer" strings followed by 24 "Winter" strings.

    seasons = (SUMMER_STRING, WINTER_STRING)  # creating a 48 item long array of 24 "Summer" strings followed by 24 "Winter" strings.

    # seasons_full = np.tile(seasons, reps=int(data_length//48))  # creating the full length of the seasons thingy.
    seasons_full = np.tile(seasons, reps=int(data_length//2))  # creating the full length of the seasons thingy.
    return seasons_full


def get_hours(data_length: int) -> np.ndarray:
    """
    :param data_length: the total length of the database
    :return: an array of the seasons for a future index in a dataframe.
    """
    return np.tile(np.arange(start=0, stop=23, step=1), reps=int(data_length//24))


def get_indies_array(azi_interval: int|float = 45, angle_interval: int|float = 5) -> np.ndarray:
    """
    :param azi_interval: the azimuth step interval for the database.
    :param angle_interval: the angle step interval for the database.
    :return: the empty dataframe with [season, azimuth, angle, hours] that later act as indexes.
    """
    # azi_steps = int(360//azi_interval)
    # angle_steps = int(90//angle_interval)
    # data_length = int(48*azi_steps*angle_steps + 48)

    azi_steps = int(360//azi_interval+1)
    angle_steps = int(90//angle_interval+1)
    data_length = int(azi_steps*angle_steps*2)
    
    seasons_full = get_seasons(data_length)
    azimuth_temp = get_azimuth_array(azi_interval, angle_steps, data_length)
    angle_temp = get_angle_array(angle_interval, azi_steps, data_length)
    # hours = get_hours(data_length=data_length)

    # df_temp = np.empty(shape=4, dtype=np.ndarray)  # [season, azimuth, angle, hour]
    # df_temp[0:3] = (seasons_full, azimuth_temp, angle_temp, hours)
    df_temp = np.empty(shape=3, dtype=np.ndarray)  # [season, azimuth, angle, hour]
    df_temp[0:3] = (seasons_full, azimuth_temp, angle_temp)

    return df_temp


def get_df_values(azi_interval: int|float = 45, angle_interval: int|float = 5) -> np.ndarray:
    azi_steps = int(360//azi_interval)
    angle_steps = int(90//angle_interval)
    data_length = int(48*azi_steps*angle_steps + 48)
    
    mod_azi = 180.  # south
    angle = angle_interval
    values = np.empty(shape=data_length, dtype=float)
    index = 0

    values[0:48] = get_daily_values(180, 0)  # the flat case.
    index += 48

    for i in range(azi_steps):  # iterating over the azimuth angles.
        for j in range(angle_steps):  # iterating over the module tilt angles
            values[index: index+48] = get_daily_values(mod_angle=angle, mod_azimuth=mod_azi)
            angle += angle_interval
            index += 48
        mod_azi -= azi_interval
        angle = angle_interval
    return values


def create_dbs_solarvals(azi_interval: int|float = 45, angle_interval: int|float = 5) -> None:
    """
    creates databases for different values for both summer and winter.
    Creates values for south, west, east and north.
    For each one, create values for each 10 degree step from 10 to 90. and one for a flat one since there is no compass direction then.
    [azimuth, angle, hour, average for summer/average for winter in percent from 0 to 1 of full sunlight] so 24 values per entry. 48 for both summer and winter.
    9 times per compass direction and 4 compass directions. Rest can be interpolated between them. Overall 1728 rows of data.
    [season, azimuth, angle, hour, percent]

    module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
    """
    azi_steps = int(360//azi_interval)
    angle_steps = int(90//angle_interval)
    data_length = int(48*azi_steps*angle_steps + 48)
    
    mod_azi = 180.  # south
    angle = angle_interval
    first = True
    
    indies = get_indies_array(azi_interval=azi_interval, angle_interval=angle_interval)
    values = get_df_values(azi_interval, angle_interval)

    index = pd.MultiIndex.from_arrays(indies, names=INDEX_NAMES)
    df = pd.DataFrame(index=index, data=values, columns=["AVG_Percent"])
    path = DIR_PATH + "/Databases/Helper/SolarPCTG.csv"
    df.to_csv(path_or_buf=path, sep=DELIMITER)


def get_dayofyear_range_array(start_day: int, end_day:int, isLeapYear:bool = False) -> np.ndarray:
    """
    :args: pretty obvious.
    :return: an array of the days of year between start_day and end_day.
    """
    last_day = 365
    if start_day < 1 or start_day > last_day or end_day < 1 or end_day > last_day:
        raise ValueError("Days are out of range for a year. You might need to find a different service for your planet.")
    
    if isLeapYear:
        last_day = 366

    if end_day < start_day:  # if it goes over jan 1st.
        return np.concatenate((np.arange(start=start_day, stop=last_day+1, step=1), np.arange(start=1, stop=end_day, step=1)))
    
    # if it doesn't go over jan 1st.
    return np.arange(start=start_day, stop=end_day+1, step=1)


def get_average_winter(mod_azimuth: float|int, mod_angle: float|int, solar_db: pd.DataFrame, summer: bool = False) -> np.ndarray:
    """
    :param mod_azimuth: the azimuth of the module. module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180. 
    Acceptable values between -180 and 180.
    :param mod_angle: the module angle from 0 to 90. If you wish to install a module at a different angle, visit a doctor, you need help.
    :param solar_db: the solar database with [DayofYear, Hour, Diffuse_Power, Global_Power] the powers are in W.

    :return: the winter averages for percent of total radiation that reaches the dumbass modules.
    """
    if summer:
        start_day = pd.to_datetime(datetime(year=2022, month=4, day=1)).dayofyear  # starting 2022 because we go over a january 1st.
        end_day = pd.to_datetime(datetime(year=2023, month=9, day=30)).dayofyear
        sol_rad_values = solar_db.loc[start_day:end_day]
        season = "summer"
        days = np.repeat(get_dayofyear_range_array(start_day=start_day, end_day=end_day), repeats=24)
    else:
        start_day = pd.to_datetime(datetime(year=2022, month=10, day=1)).dayofyear  # starting 2022 because we go over a january 1st.
        end_day = pd.to_datetime(datetime(year=2023, month=3, day=31)).dayofyear

        sol_rad_values = pd.concat([solar_db.loc[start_day:], solar_db.loc[1:end_day]])
        season = "winter"
        days = np.repeat(get_dayofyear_range_array(start_day=start_day, end_day=end_day+1), repeats=24)
    
    # location = PVLocation(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE)
    # pvsys = PVSystem(surface_tilt=mod_angle, surface_azimuth=mod_azimuth, albedo=ALBEDO)
    # modelchain = ModelChain(system=pvsys, location=location)
    # modelchain.run_model()

    
    times = np.tile(np.arange(24), reps=int(len(days)/24))/24  # dividing by 24 because my nonsense time goes from 0 to 1.
    diffuse_rad = sol_rad_values[DIFF_STRING].to_numpy(dtype=float)
    global_rad = sol_rad_values[GLOB_STRING_ALT].to_numpy(dtype=float)

    rad_onmodule_values = get__array_rad_onmodule(days=days, times=times, diff_rads=diffuse_rad, total_rads=global_rad, module_angles=mod_angle, module_azimuths=mod_azimuth, timezone=1)
    total_global_radiation = np.sum(global_rad)/1000  # in kWh. not divided by six because hourly steps here.
    total_radiation_on_module = np.sum(rad_onmodule_values)/1000  # in kWh. not divided by six because hourly steps here.
    print(f"For a module with an azimuth of {AZIMUTH_STRING} and angle of {mod_angle} In the {season}, the total global is: {round(total_global_radiation, 0)} kWh")
    print(f"And the total on module is: {round(total_radiation_on_module, 0)} kWh")
    
    times = np.tile(np.arange(24), reps=int(len(days)/24))
    index = pd.MultiIndex.from_arrays((days, times), names=[DAY_OF_YEAR_STRING, TIME_STRING])
    data = pd.array([global_rad, rad_onmodule_values])

    df = pd.DataFrame(index=index, columns=[GLOB_STRING_ALT, RAD_ON_MOD_STRING])
    df[GLOB_STRING_ALT] = global_rad
    df[RAD_ON_MOD_STRING] = rad_onmodule_values
    # df[TIME_STRING] = df.index.get_level_values(TIME_STRING)
    df = df.groupby(TIME_STRING).mean()
    df[PERCENT_STRING] = df[RAD_ON_MOD_STRING]/1000
    # df = df.fillna(0)

    return df[PERCENT_STRING].to_numpy()


def get_poa(solar_db: pd.DataFrame, location: PVLocation, surface_tilt: float|int, surface_azi: float|int) -> pd.DataFrame:
    global TEMP_AIR_STRING, WIND_SPEED_STRING, GLOBAL_STRING, DIFF_HOR_STRING, DIFF_NORM_STRING
    # times = solar_db.index
    times = pd.to_datetime(solar_db.index.values)
    solarpos = location.get_solarposition(times=times)
    poa_data = pvlib.irradiance.get_total_irradiance(surface_tilt=surface_tilt, surface_azimuth=surface_azi,
                                                dni= solar_db[DIFF_NORM_STRING],
                                                ghi= solar_db[GLOBAL_STRING],
                                                dhi=solar_db[DIFF_HOR_STRING],
                                                solar_zenith=solarpos["apparent_zenith"],
                                                solar_azimuth=solarpos["azimuth"],
                                                )
    poa_data[TEMP_AIR_STRING] = solar_db[TEMP_AIR_STRING]
    poa_data[WIND_SPEED_STRING] = solar_db[WIND_SPEED_STRING]
    return poa_data


def get_poa_season(is_summer: bool, solar_db: pd.DataFrame, location:PVLocation, surface_tilt: float|int, surface_azi: float|int) -> pd.DataFrame:
    if is_summer:
        year_db = pd.to_datetime(solar_db.index.values[0]).year
        start_day = pd.to_datetime(datetime(year=year_db, month=4, day=1))
        end_day = pd.to_datetime(datetime(year=year_db, month=9, day=30))
        sol_rad_values = solar_db.loc[start_day:end_day]
    else:
        year_db = pd.to_datetime(solar_db.index.values[0]).year
        start_day = pd.to_datetime(datetime(year=year_db, month=10, day=1))
        end_day = pd.to_datetime(datetime(year=year_db, month=3, day=31))

        sol_rad_values = pd.concat([solar_db.loc[start_day:], solar_db.loc[:end_day]])

    return get_poa(solar_db, location, surface_tilt, surface_azi)


def tester_winter(mod_azimuth: float|int, mod_angle: float|int, solar_db: pd.DataFrame) -> np.ndarray:
    """
    :param mod_azimuth: the azimuth of the module. module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180. 
    Acceptable values between -180 and 180.
    :param mod_angle: the module angle from 0 to 90. If you wish to install a module at a different angle, visit a doctor, you need help.
    :param solar_db: the solar database with [DayofYear, Hour, Diffuse_Power, Global_Power] the powers are in W.

    :return: the winter averages for percent of total radiation that reaches the dumbass modules.
    """
    # https://static.viessmann.com/resources/technical_documents/DE/de/VDP/6220869VDP00002_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit
    celltype = "monoSi"  # type = monokristallin
    pdc0 = 445  # Nennleistung in W
    v_mp = 33.22  # Optimale operative Spannung in V
    i_mp = 13.4  # Optimaler operativer Strom in A
    v_oc = 39.2  # Leerlaufspannung in V
    i_sc = 14.19  # Kurzschlussstrom in A
    alpha_sc = 0.00043*i_sc  # Temperaturkoeffizient Kurzschlussstrom 0,043 %/°C
    beta_voc = -0.0024*v_oc  # Temperaturkoeffizient Leerlaufspannung -0,24 %/°C
    gamma_pdc = -0.003  # Temperaturkoeffizient maximale Leistung -0,3 %/°C
    cells_in_series = 22  # 22 Module für 9,79 kWp
    temp_ref = 25  # Referenztemperatur 25 °C

    year_db = pd.to_datetime(solar_db.index.values[1]).year

    start_day = pd.to_datetime(datetime(year=year_db, month=10, day=1))
    end_day = pd.to_datetime(datetime(year=year_db, month=3, day=31))
    
    sol_rad_values = pd.concat([solar_db.loc[start_day:], solar_db.loc[:end_day]])
    location = PVLocation(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE)
    poa_data = get_poa(sol_rad_values, location, mod_angle, mod_azimuth)
    effective_irradiance = poa_data["poa_direct"] + poa_data["poa_diffuse"]

    temp_cell = pvlib.temperature.faiman(poa_data["poa_global"], poa_data[TEMP_AIR_STRING])

    result_dc = pvlib.pvsystem.pvwatts_dc(effective_irradiance,  temp_cell, pdc0*cells_in_series, gamma_pdc, temp_ref=temp_ref)
    
    # https://static.viessmann.com/resources/technical_documents/DE/de/VDP/4484027VDP00001_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit 
    result_ac = (pvlib.inverter.pvwatts(result_dc, pdc0=20000, eta_inv_nom=0.975, eta_inv_ref=0.968))/(pdc0*cells_in_series)

    df_results = pd.DataFrame(result_ac)
    df_results.index = pd.to_datetime(df_results.index)
    df_results[TIME_STRING] = df_results.index.time

    df_results = df_results.groupby(TIME_STRING).mean()

    return np.ravel(df_results.values)


def test_a_few_winters():
    directions = {-180: "South", -135: "Southwest", -90: "West", -45:"Northwest", 0: "North",  45: "Northeast", 90: "East", 135: "Southeast", 180: "South"}
    keys = list(directions.keys())[:-1]
    print
    solar_db = test_solar_db()
    """
    Flat case should be the same for all fuckers.
    """
    # mod_azi = 0
    # mod_angle = 0
    # tester_winter(mod_azi, mod_angle, solar_db)
    for i in range(len(directions.keys())):
        mod_azi = keys[i]
        for i in range(10):
            mod_angle = i*10
            tester_winter(mod_azi, mod_angle, solar_db)
        # for i in range(9):
        #     mod_angle = (i+1)*10


def create_df_seasons():
    """
    The master dawg. Here we test it all.
    """
    directions = {-180: "South", -135: "Southwest", -90: "West", -45:"Northwest", 0: "North",  45: "Northeast", 90: "East", 135: "Southeast", 180: "South"}
    azi_keys = list(directions.keys())
    num_dir = len(directions.keys())
    solar_db = test_solar_db()
    azi_step = 45
    angle_step = 5
    if 90 % angle_step == 0:
        angles = np.arange(0, 91, angle_step)
    else:
        angles = np.arange(0, 90+angle_step, angle_step)
        angles[-1] = 90
    # index_keys = np.repeat(keys, num_angles)
    # index_angles = np.repeat(np.arange(0, 91, angle_step), num_dir)
    # index = pd.MultiIndex.from_arrays((index_keys, index_angles), names=[AZIMUTH_STRING, ANGLE_STRING])

    index = pd.MultiIndex.from_arrays(get_indies_array(azi_step, angle_step), names=[SEASON_STRING, AZIMUTH_STRING, ANGLE_STRING])
    ret_df = pd.DataFrame(index=index)
    ret_df[PERCENT_STRING] = np.empty((index.shape[0])).astype("object")

    """
    Flat case should be the same for all fuckers.
    """
    mod_azi = 0
    mod_angle = 0
    flat_vals_summer = tester_summer(mod_azi, mod_angle, solar_db).tolist()
    flat_vals_winter = tester_winter(mod_azi, mod_angle, solar_db).tolist()
    for mod_azi in azi_keys:
        mod_angle = 0
        ret_df.at[(SUMMER_STRING, mod_azi, mod_angle), PERCENT_STRING] = flat_vals_summer
        ret_df.at[(WINTER_STRING, mod_azi, mod_angle), PERCENT_STRING] = flat_vals_winter
    for mod_azi in azi_keys:
        for mod_angle in angles:
            ret_df.at[(SUMMER_STRING, mod_azi, mod_angle), PERCENT_STRING] = tester_summer(mod_azi, mod_angle, solar_db).tolist()
            ret_df.at[(WINTER_STRING, mod_azi, mod_angle), PERCENT_STRING] = tester_winter(mod_azi, mod_angle, solar_db).tolist()
        # for i in range(9):
        #     mod_angle = (i+1)*10
    
    path_sol = DIR_PATH + f"/Databases/Seasons/PCTs.csv"
    ret_df.to_csv(path_sol, sep=";")


def tester_summer(mod_azimuth: float|int, mod_angle: float|int, solar_db: pd.DataFrame) -> np.ndarray:
    """
    :param mod_azimuth: the azimuth of the module. module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180. 
    Acceptable values between -180 and 180.
    :param mod_angle: the module angle from 0 to 90. If you wish to install a module at a different angle, visit a doctor, you need help.
    :param solar_db: the solar database with [DayofYear, Hour, Diffuse_Power, Global_Power] the powers are in W.

    :return: the summer averages for percent of total radiation that reaches the dumbass modules.
    """
    # https://static.viessmann.com/resources/technical_documents/DE/de/VDP/6220869VDP00002_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit
    celltype = "monoSi"
    pdc0 = 445
    v_mp = 33.22
    i_mp = 13.4
    v_oc = 39.2
    i_sc = 14.19
    alpha_sc = 0.00043*i_sc
    beta_voc = -0.0024*v_oc
    gamma_pdc = -0.003
    cells_in_series = 22
    temp_ref = 25

    year_db = pd.to_datetime(solar_db.index.values[1]).year

    start_day = pd.to_datetime(datetime(year=year_db, month=4, day=1))
    end_day = pd.to_datetime(datetime(year=year_db, month=9, day=30))
    
    sol_rad_values = solar_db.loc[start_day:end_day]
    location = PVLocation(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE)
    poa_data = get_poa(sol_rad_values, location, mod_angle, mod_azimuth)
    effective_irradiance = poa_data["poa_direct"] + poa_data["poa_diffuse"]

    temp_cell = pvlib.temperature.faiman(poa_data["poa_global"], poa_data[TEMP_AIR_STRING])

    result_dc = pvlib.pvsystem.pvwatts_dc(effective_irradiance,  temp_cell, pdc0*cells_in_series, gamma_pdc, temp_ref=temp_ref)
    
    # https://static.viessmann.com/resources/technical_documents/DE/de/VDP/4484027VDP00001_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit 
    result_ac = (pvlib.inverter.pvwatts(result_dc, pdc0=20000, eta_inv_nom=0.975, eta_inv_ref=0.968))/(pdc0*cells_in_series)

    df_results = pd.DataFrame(result_ac)
    df_results.index = pd.to_datetime(df_results.index)
    df_results[TIME_STRING] = df_results.index.time

    df_results = df_results.groupby(TIME_STRING).mean()

    return np.ravel(df_results.values)


def tester_summer_comments():
    pass
    # start_day = pd.to_datetime(datetime(year=year_db, month=4, day=1)).tz = TIMEZONE
    # end_day = pd.to_datetime(datetime(year=year_db, month=9, day=30)).tz = TIMEZONE
    # times = pd.date_range(start_day, freq="1h" , end=end_day , tz=TIMEZONE)

    # sol_rad_values = solar_db.loc[times]
    # temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    # sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    # sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    # cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    # solarpos = location.get_solarposition(times=pd.to_datetime(sol_rad_values.index.values))
    # aoi = pvlib.irradiance.aoi(mod_angle, mod_azimuth, solarpos["apparent_zenith"], solarpos["azimuth"])
    # iam = pvlib.iam.ashrae(aoi)
    # effective_irradiance = poa_data["poa_direct"]*iam + poa_data["poa_diffuse"]
    # result_dc.plot(figsize=(16,9))
    # plt.show()

    # plt.ylabel("PV Power as percent of nominal power")
    # plt.xlabel("Time of year")
    # plt.title(f"Summer average power for module angle of {mod_angle} and direction {direction}")
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    # result_ac.plot(figsize=(16,9))

    # df_results = pd.DataFrame(result_ac)
    # df_results.index = pd.to_datetime(df_results.index)
    # df_results[TIME_STRING] = df_results.index.time

    # df_results = df_results.groupby(TIME_STRING).mean()
    # plt.show()

    # df_results.plot(figsize=(16,9))
    # plt.ylabel("PV Power as percent of nominal power")
    # plt.xlabel("Time of day")

    # plt.title(f"Summer average power of average day for module angle of {mod_angle} and direction {direction}")
    # plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    # plt.show()

    # azi = (mod_azimuth+180)%360-180
    # if azi != 0:
    #     azi = round(azi.__abs__()/45+0.01, 0)*45*azi.__abs__()/azi
    
    # directions = {-180: "South", -135: "Southwest", -90: "West", -45:"Northwest", 0: "North",  45: "Northeast", 90: "East", 135: "Southeast", 180: "South"}

    # direction = directions[azi]

    # pvsys = PVSystem(surface_tilt=mod_angle, surface_azimuth=mod_azimuth, albedo=ALBEDO, module_parameters=sandia_module, inverter_parameters=cec_inverter,
    #                  temperature_model_parameters=temperature_model_parameters,
    #                  modules_per_string=7, strings_per_inverter=2)
    # modelchain = ModelChain(system=pvsys, location=location)
    # modelchain.run_model(sol_rad_values)
    # modelchain.results.ac.plot(figsize=(16, 9))
    # plt.show()
    

def get_average_summer(mod_azimuth: float|int, mod_angle: float|int, solar_db: pd.DataFrame) -> np.ndarray:
    start_day = pd.to_datetime(datetime(year=2023, month=4, day=1)).day_of_year
    end_day = pd.to_datetime(datetime(year=2023, month=9, day=30)).dayofyear

    sol_rad_values = solar_db.loc[start_day:end_day]

    days = np.repeat(get_dayofyear_range_array(start_day=start_day, end_day=end_day+1), repeats=24)
    times = np.tile(np.arange(24), reps=int(len(days)/24))/24  # dividing by 24 because my nonsense time goes from 0 to 1.
    diffuse_rad = sol_rad_values[DIFF_STRING].to_numpy(dtype=float)
    global_rad = sol_rad_values[GLOB_STRING_ALT].to_numpy(dtype=float)

    rad_onmodule_values = get__array_rad_onmodule(days=days, times=times, diff_rads=diffuse_rad, total_rads=global_rad, module_angles=mod_angle, module_azimuths=mod_azimuth, timezone=1)
    total_global_radiation = np.sum(global_rad)/1000  # in kWh. not divided by six because hourly steps here.
    total_radiation_on_module = np.sum(rad_onmodule_values)/1000  # in kWh. not divided by six because hourly steps here.
    
    times = np.tile(np.arange(24), reps=int(len(days)/24))
    index = pd.MultiIndex.from_arrays((days, times), names=[DAY_OF_YEAR_STRING, TIME_STRING])
    data = pd.array([global_rad, rad_onmodule_values])

    df = pd.DataFrame(index=index, columns=[GLOB_STRING_ALT, RAD_ON_MOD_STRING])
    df[GLOB_STRING_ALT] = global_rad
    df[RAD_ON_MOD_STRING] = rad_onmodule_values
    # df[TIME_STRING] = df.index.get_level_values(TIME_STRING)
    df = df.groupby(TIME_STRING).mean()
    df[PERCENT_STRING] = df[RAD_ON_MOD_STRING]/1000
    # df = df.fillna(0)

    return df[PERCENT_STRING].to_numpy()


def get_daily_values(mod_azimuth: float|int, mod_angle: float|int) -> np.ndarray:
    """
    Start summer then winter.
    """
    solar_db = get_solar_rad_db()
    values = np.empty(48)
    values[0:24] = get_average_winter(mod_azimuth=mod_azimuth, mod_angle=mod_angle)
    values[24:48] = get_average_summer(mod_azimuth=mod_azimuth, mod_angle=mod_angle)
    
    return values


def get_solar_rad_db(day_interval:int = 1, fullhour = True) -> pd.DataFrame:
    """
    gives back the database for the solar radiation data with the interval day_interval. #
    Can use 7 to get representative weekly values and speed up calculations.
    :param day_interval: the day interval of values returned.
    """
    df = pd.read_csv(filepath_or_buffer=SOLPATH, delimiter=DELIMITER, names=(DATETIME_STRING, GLOB_STRING_ALT, DIFF_STRING, DIFF_NORM_STRING, WIND_SPEED_STRING, TEMP_AIR_STRING), header=0)
    df = df.set_index(DATETIME_STRING, drop=True)
    df.index = pd.to_datetime(df.index)
    na = df.isna().sum()
    df = df.resample("h").mean()
    df[DATETIME_STRING] = df.index
    df[DATETIME_STRING] = pd.to_datetime(df[DATETIME_STRING])
    df[DAY_OF_YEAR_STRING] = df[DATETIME_STRING].dt.dayofyear
    df[TIME_STRING] = df[DATETIME_STRING].dt.time
    df = df[[DAY_OF_YEAR_STRING, TIME_STRING, DIFF_STRING, GLOB_STRING_ALT]]

    df = df.set_index([DAY_OF_YEAR_STRING, TIME_STRING], drop=True)
    days = np.arange(start=1, stop=366, step=day_interval)
    df = df.loc[days]
    # arie = df.to_numpy()
    # first_thingy= arie[0][0]
    # print(first_thingy)
    # # 2022-06-20 00:00:00
    # # 
    # print(type(first_thingy))
    return df
    

def test_solar_db() -> pd.DataFrame:
    """
    gives back the database for the solar radiation data with the interval day_interval. #
    Can use 7 to get representative weekly values and speed up calculations.
    :param day_interval: the day interval of values returned.
    """
    
    global GLOBAL_STRING, DIFF_HOR_STRING, DIFF_NORM_STRING, WIND_SPEED_STRING, TEMP_AIR_STRING
    df = pd.read_csv(filepath_or_buffer=SOLPATH, delimiter=DELIMITER)
    df = df.set_index(DATETIME_STRING, drop=True)
    df.index = pd.to_datetime(df.index)
    na = df.isna().sum()
    df = df.resample("h").mean()
    df = df[[GLOBAL_STRING, DIFF_HOR_STRING, DIFF_NORM_STRING, TEMP_AIR_STRING, WIND_SPEED_STRING]]

    return df


def get_clear_sky(year):
    global LATITUDE, LONGITUDE
    date = f"01-01-{year}"
    site_location = pvlib.location.Location(LATITUDE,LONGITUDE, TIMEZONE)
    # Creates one day's worth of 10 min intervals
    times = pd.date_range(date, freq='1h', periods=24*365,
                          tz=site_location.tz)
    
    solar_db = test_solar_db()
    # Generate clearsky data using the Ineichen model, which is the default
    # The get_clearsky method returns a dataframe with values for GHI, DNI,
    # and DHI
    clearsky = site_location.get_clearsky(times)
    
    clearsky[WIND_SPEED_STRING] = solar_db[WIND_SPEED_STRING].values
    clearsky[TEMP_AIR_STRING] = solar_db[TEMP_AIR_STRING].values
    return clearsky



solar_db = get_solar_rad_db()

# solar_db = test_solar_db()
# tester_winter(0,90, solar_db)

# test_a_few_winters()
# create_df_seasons()

get_average_winter(180, 35, solar_db)
get_average_winter(180, 0, solar_db)
get_average_winter(180, 35, solar_db, 1)
get_average_winter(180, 0, solar_db, 1)
# get_solar_rad_db(1)
# azi_interval = 45
# angle_interval = 5
# azi_steps = int(360//azi_interval)
# angle_steps = int(90//angle_interval)
# data_length = int(48*azi_steps*angle_steps + 48)
# print(get_angle_array(angle_interval=angle_interval, azi_steps=azi_steps, data_length=data_length))
