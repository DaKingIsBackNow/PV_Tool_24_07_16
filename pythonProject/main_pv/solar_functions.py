import typing
import numpy as np
from datetime import datetime
import pandas as pd
from pvlib.location import Location as PVLocation
import pvlib

from main_pv.solar_validations import *
from main_pv.standard_variables import *

""" 
Functions for solar radiation power calculations.  Use these to determine  how much 
electrical power could be generated from a photovoltaic module in a specified  location 
on a particular day, given radiation values.
"""


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


def get_pctgs(mod_azimuth: float|int, mod_angle: float|int, solar_db: pd.DataFrame) -> np.ndarray:
    """
    :param mod_azimuth: the azimuth of the module. module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180. 
    Acceptable values between -180 and 180.
    :param mod_angle: the module angle from 0 to 90. If you wish to install a module at a different angle, visit a doctor, you need help.
    :param solar_db: the solar database with [DayofYear, Hour, Diffuse_Power, Global_Power] the powers are in W.

    :return: the averages for percent of total radiation that reaches the dumbass modules.
    """
    global celltype, pdc0, v_mp, i_mp, v_oc, i_sc, alpha_sc, beta_voc, gamma_pdc, cells_in_series, temp_ref, inv_pdc0, eta_inv_nom, eta_inv_ref
    solar_db.index = pd.to_datetime(solar_db.index)

    year_db = pd.to_datetime(solar_db.index.values[1]).year

    start_day = pd.to_datetime(datetime(year=year_db, month=1, day=1))
    end_day = pd.to_datetime(datetime(year=year_db+1, month=1, day=1))
    
    sol_rad_values = solar_db.loc[start_day:end_day]
    location = PVLocation(latitude=LATITUDE, longitude=LONGITUDE, tz=TIMEZONE)

    poa_data = get_poa(sol_rad_values, location, mod_angle, mod_azimuth)
    effective_irradiance = poa_data["poa_direct"] + poa_data["poa_diffuse"]

    temp_cell = pvlib.temperature.faiman(poa_data["poa_global"], poa_data[TEMP_AIR_STRING])

    result_dc = pvlib.pvsystem.pvwatts_dc(effective_irradiance,  temp_cell, pdc0*cells_in_series, gamma_pdc, temp_ref=temp_ref)
    
    # https://static.viessmann.com/resources/technical_documents/DE/de/VDP/4484027VDP00001_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit 
    result_ac = (pvlib.inverter.pvwatts(result_dc, pdc0=inv_pdc0, eta_inv_nom=eta_inv_nom, eta_inv_ref=eta_inv_ref))/(pdc0*cells_in_series)
    # dividing by max power to get percent of max power

    df_results = pd.DataFrame(result_ac)

    return np.ravel(df_results.values)


def fix_time(time:np.ndarray) -> np.ndarray:
    """
    Time needs to be between 0 and 1.
    """
    return (time + 1) % 1


def get_len_float_int_array(value: float|int|np.ndarray) -> int:
    try:
        return len(value)
    except TypeError:  # if it's an int or float
        return 1


def get_percent_onmodule(datey_time: datetime, module_angle: float|int = 30, module_azimuth: float|int = 180, longitude: float = 9.18, latitude: float = 48.78, timezone: int = 1, 
                        total_rad: float = 1000, diff_rad: float = None):
    """
    Apply a mathematical function to an angle given in degrees.

    Args:
            :param datey_time: the current datetime.
            :param module_angle: (callable): The dumbass module angle.
            :param module_azimuth: module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
            :param longitude: the longitude, Standard is Stuttgart.
            :param latitude: the latitude standard is Stuttgart.
            :param timezone: the timezone +1 for Germany winter time.

        Returns:
            float: the percent of global radiation that falls on the module.
    """
    if diff_rad is None:
        diff_rad = total_rad*0.4

    day_num = (datey_time - datetime(datey_time.year, 1, 1)).days - 1
    par_j = get_par_j(day_num)
    sun_dec = get_sun_declination(par_j)
    time_parameter = get_time_parameter_in_days(par_j)
    loc_time = datey_time.hour/24 + datey_time.minute/60/24
    timezone = timezone/24
    moz = get_mean_loc_time(loc_time, timezone, longitude)
    woz = get_true_local_time(moz, time_parameter)
    hour_angle = get_hour_angle(woz)
    sol_alt = get_solar_altitude(hour_angle, latitude, sun_dec)
    sol_azim = get_solar_azimuth(sol_alt, latitude, sun_dec, woz)
    sun_vec = get_sun_vector(sol_azim, sol_alt)
    module_vec = get_module_vector(module_azimuth, module_angle)

    indecent_angle = get_angle_incidence(sun_vec, module_vec)

    dir_rad_mod = get_direct_rad_module(total_rad-diff_rad, indecent_angle, sol_alt)
    diff_rad_mod = get_diffuse_rad_module(diff_rad, module_angle)
    ref_rad_mod = get_reflect_rad_module(total_rad, module_angle)

    return (dir_rad_mod + diff_rad_mod + ref_rad_mod)/total_rad


def get__array_rad_onmodule(days: np.ndarray, times: np.ndarray, total_rads: np.ndarray, diff_rads: np.ndarray, \
                                module_angles: float|int|np.ndarray = 30, module_azimuths: float|int|np.ndarray = 180, \
                                longitude: float = 9.18, latitude: float = 48.78, timezone: int = 1) -> np.ndarray:
    """
    Apply a mathematical function to an angle given in degrees.

    Args:
            :param datey_time: the current datetime.
            :param module_angle: The dumbass module angle.
            :param module_azimuth: module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
            :param longitude: the longitude, Standard is Stuttgart.
            :param latitude: the latitude standard is Stuttgart.
            :param timezone: the timezone +1 for Germany winter time. +2 for summer time.

        Returns:
            float: the percent of global radiation that falls on the module.
    """
    data_length = len(days)
    len_angles = get_len_float_int_array(module_angles)
    len_azimuths = get_len_float_int_array(module_azimuths)

    if len_angles == 1:
        len_angles = data_length
    if len_azimuths == 1:
        len_azimuths = data_length

    if not len(days) == len_angles == len_azimuths == len(times) == len(total_rads) == len(diff_rads):
        raise ValueError("The dataframes aren't the same length. check your stuff dawg. (datetimes, module angles, module azimuths)")
    
    timezone = timezone/24
    pars_j = get_par_j(days)
    sun_decs = get_sun_declination(pars_j)
    time_parameters = get_time_parameter_in_days(pars_j)
    loc_times = times

    mozs = get_mean_loc_time(loc_times, timezone, longitude)
    wozs = get_true_local_time(mozs, time_parameters)
    hour_angles = get_hour_angle(wozs)

    sol_alts = get_solar_altitude(hour_angles, latitude, sun_decs)
    sol_azims = get_solar_azimuth(sol_alts, latitude, sun_decs, wozs)
    sun_vecs = get_sun_vector(sol_azims, sol_alts)
    module_vecs = get_module_vector(module_azimuths, module_angles)

    indecent_angles = get_angle_incidence(sun_vecs, module_vecs)

    # multi = (np.matmul(sun_vecs.T[200], module_vecs))
    # print(multi)

    dir_rad_mods = get_direct_rad_module(total_rads-diff_rads, indecent_angles, sol_alts)
    diff_rad_mods = get_diffuse_rad_module(diff_rads, module_angles)
    ref_rad_mods = get_reflect_rad_module(total_rads, module_angles)

    return (dir_rad_mods + diff_rad_mods + ref_rad_mods)


def get_dni(day_num:np.ndarray, times: np.ndarray[datetime], ghi:np.ndarray, dhi:np.ndarray, longitude: float = 9.18, latitude: float = 48.78, timezone: int = 1) -> np.ndarray:
    """
    Gives you back the direct normal irradiance using the magic of trigonometry.
    Args:
            :param day_num: the day of year from 1 to 365, ignoring leap years. Leap years don't exist in our universe.
            :param times: Local time (0 to 1, where 0 is 0:00 and 1 is 0:00).
            :param ghi: the global radiation.
            :param dhi: the diffuse horizontal thingy.
            :param longitude: the longitude, Standard is Stuttgart.
            :param latitude: the latitude standard is Stuttgart.
            :param timezone: the timezone +1 for Germany winter time. +2 for summer time.

        Returns:
            ndarray of floats: My estimation of the direct normal irradiance. Not perfect. Hard coded numbers. But what are ya gonna do? We work with what we got. 
    """
    par_j = get_par_j(day_num)
    sun_dec = get_sun_declination(par_j)
    time_parameter = get_time_parameter_in_days(par_j)
    loc_time = times
    timezone = timezone/24
    moz = get_mean_loc_time(loc_time, timezone, longitude)
    woz = get_true_local_time(moz, time_parameter)
    hour_angle = get_hour_angle(woz)
    sol_alt = np.radians(get_solar_altitude(hour_angle, latitude, sun_dec))
    sinalt = np.clip(np.sin(sol_alt), 0.1, 1)
    return np.clip((ghi - dhi)/sinalt, 0, 1400)

def get_zeniths(day_num:np.ndarray, times: np.ndarray[datetime], longitude: float = 9.18, latitude: float = 48.78, timezone: int = 1) -> np.ndarray:
    par_j = get_par_j(day_num)
    sun_dec = get_sun_declination(par_j)
    time_parameter = get_time_parameter_in_days(par_j)
    loc_time = times
    timezone = timezone/24
    moz = get_mean_loc_time(loc_time, timezone, longitude)
    woz = get_true_local_time(moz, time_parameter)
    hour_angle = get_hour_angle(woz)
    sol_alt = get_solar_altitude(hour_angle, latitude, sun_dec)

    small_posi = np.where(np.logical_and(sol_alt>= 0, sol_alt<= 0.5)) 
    small_negi = np.where(np.logical_and(sol_alt>= -0.5, sol_alt<= 0)) 
    
    sol_alt[small_posi] += 3
    sol_alt[small_negi] -= 3

    zenith = 90-sol_alt
    return zenith

def deg_func(math_func: typing.Callable, degrees: typing.Union[int, float]|np.ndarray) -> float:
    """
    Apply a mathematical function to an angle given in degrees.

    Args:
            math_func (callable): The mathematical function to apply.
            degrees (int or float): The angle in degrees.

        Returns:
            float: The result of applying the function to the angle in radians.
    """
    try:
        radians = np.radians(degrees)
        return math_func(radians)
    except Exception as e:
        print(f"Error: {e}")
        return float('nan')  # Return NaN for invalid inputs or errors


def get_par_j(day: int|np.ndarray) -> float:
    """
    Returns the paremter J for the current day of year
    J = 360°*(Tag des Jahres/Zahl der Tage im Jahr)	[°]
    Args:
            day: the day of year from 1 to 365, ignoring leap years. Leap years don't exist in our universe.

        Returns:
            the sun declination in degrees.
    """
    validate_day(day)

    return 360 * day / 366


def get_sun_declination(par_j: float|np.ndarray):
    """
    Calculates the sun declination by time of year.
    δ = 0,3948 − 23.2559 ∙ cos(J´ + 9,1°) − 0.3915 ∙ cos(2 ∙ J´ + 5,4°) − 0.1764 ∙ cos(3 ∙ J´ + 26°)	[°]
    Args:
            par_j: porameter for time of year, where 0 is the first of january and 1 is december 31st.

        Returns:
            the sun declination in degrees.
    """

    validate_par_j(par_j)

    param_1 = 0.3948
    param_2 = 23.2559
    param_3 = 9.1
    param_4 = 0.3915
    param_5 = 5.4
    param_6 = 0.1764
    param_7 = 26
    return (param_1 - param_2 * deg_func(np.cos, (par_j + param_3)) - param_4 *
            deg_func(np.cos, (2 * par_j + param_5)) - param_6 *
            deg_func(np.cos, (3 * par_j + param_7)))


def get_time_parameter_in_days(par_j: float):
    """
    Calculates the time parameter used to shift current time to true time by time of year.
    Zgl = 0,0066 + 7,3525 ∙ cos(J´ + 85,9°) + 9,9359 ∙ cos(2 ∙ J´ + 108,9°) + 0,3387 ∙ cos(3 ∙ J´ + 105,2°)	[min]
    Args:
            par_j: porameter for time of year, where 0 is the first of january and 1 is december 31st.

        Returns:
            time parameter Zgl for the calculation in (days).
    """

    validate_par_j(par_j)

    param_1 = 0.0066
    param_2 = 7.3525
    param_3 = 85.9
    param_4 = 9.9359
    param_5 = 108.9
    param_6 = 0.3387
    param_7 = 105.2
    return (param_1 + param_2 * deg_func(np.cos, (par_j + param_3)) + param_4 *
            deg_func(np.cos, (2 * par_j + param_5)) + param_6 * deg_func(np.cos, (3 * par_j + param_7))) / 60 / 24


def get_mean_loc_time(local_time: float, timezone: float, longitude: float):
    """
        Calculates the Mean Local Time (MOZ) based on local time, timezone offset, and longitude.
        MOZ = LZ − Zeitzone + 4 ∙ λ ∙ (min/°) = LZ − Zeitzone + (4/60 )∙ λ [h/°]	[0:00-23:59]

        Args:
            local_time: Local time (0 to 1, where 0 is 0:00 and 1 is 0:00).
            timezone: Timezone offset (-1 represents 24 hours earlier,
                  +1 represents 24 hours later).
            longitude: Longitude in degrees.

        Returns:
            Mean Local Time as a float between 0 and 1.
    """

    """
    Validation
    """
    validate_loc_time(local_time)
    validate_timezone(timezone)
    validate_longitude(longitude)
    """
    End Validation. Calculation starts.
    """

    mean_local_time = local_time - timezone + (4 / 60 / 24) * longitude

    mean_local_time = fix_time(mean_local_time)

    return mean_local_time


def get_true_local_time(mean_loco_time: float, time_parameter: float):
    """
    Calculates the True Local Time based on mean local time, and time parameter.

    WOZ = MOZ + Zgl	[0:00-23:59]
        Args:
            mean_loco_time: Mean local time (0 to 1, where 0 is 0:00 and 1 is 0:00).
            time_parameter: Time parameter in days.

        Returns:
            True Local Time as a float between 0 and 1.
    """
    """
        Validation
    """
    validate_mean_loc_time(mean_loco_time)
    validate_time_parameter(time_parameter)
    """
        End of validation
    """

    real_local_time = mean_loco_time + time_parameter
    real_local_time = fix_time(real_local_time)

    return real_local_time


def get_hour_angle(true_loco_time: float):
    """
    Calculates the Hour angle in degrees.

    ω = (12h − WOZ) × (15°/h)
    ω = (12h/24h*d − WOZ/24h*d) × (15°/h)*(24h/d)
    ω = (0.5 − WOZ_in_days) × (15°/h)*(24h/d)
        Args:
            true_loco_time: True local time ajusted for longitude and time of year
            (0 to 1, where 0 is 0:00 and 1 is 0:00).

        Returns:
            Hour angle in degrees.
    """

    validate_true_loc_time(true_loco_time)

    return (0.5 - true_loco_time) * 15 * 24


def get_solar_altitude(hour_angley: float, latitude: float, sun_declination: float):
    """
    Calculates the solar altitude in degrees.

    γs = arcsin(cos ω ∙ cos φ ∙ cos δ + sin φ ∙ sin δ)	[°]
        Args:
            hour_angley: hour angle in degreees based on real local time.
            latitude: latitude in degrees.
            sun_declination: sun declination in degrees

        Returns:
            Hour angle in degrees.
    """

    """
        Validation
    """
    validate_hour_angle(hour_angley)
    validate_sun_declination(sun_declination)
    validate_latitude(latitude)
    """
    End Validation. Calculation starts.
    """
    cos_h_a = deg_func(np.cos, hour_angley)
    cos_lat = deg_func(np.cos, latitude)
    cos_s_d = deg_func(np.cos, sun_declination)
    sin_lat = deg_func(np.sin, latitude)
    sin_s_d = deg_func(np.sin, sun_declination)

    return np.degrees(np.arcsin(cos_h_a * cos_lat * cos_s_d + sin_lat * sin_s_d))


def get_solar_azimuth(sol_altitude: float, latitude: float, sun_declination: float, true_loco_time: float):
    """
        Calculates the solar azimuth in degrees.
        North = 0°, West = -90°, East = +90°, South = 180

        αs = {     180° − arccos (sin γs∙sin φ−sin δ/(cos γs∙cos φ)) wenn WOZ ≤ 12: 00 Uhr
                   180° + arccos (sin γs∙sin φ−sin δ/(cos γs∙cos φ)) wenn WOZ > 12: 00 Uhr
             }


            Args:
                sol_altitude: solar altitude in degreees based on hour angle.
                latitude: latitude in degrees.
                sun_declination: sun declination in degrees.
                true_loco_time: true local time from 0 to 1.

            Returns:
                solar azimuth in degrees.
    """

    """
        Validation
    """
    validate_sol_altitude(sol_altitude)
    validate_sun_declination(sun_declination)
    validate_latitude(latitude)
    validate_true_loc_time(true_loco_time)
    """
        End Validation. Calculation starts.
    """

    """
    Convert to radians.
    """
    sol_altitude = np.radians(sol_altitude)
    sun_declination = np.radians(sun_declination)
    latitude = np.radians(latitude)

    """
    Case for before or after noon.
    """
    bool_check = np.float16(true_loco_time < 0.5)
    addition = np.degrees(np.arccos(np.sin(sol_altitude) * np.sin(latitude) -
                      np.sin(sun_declination) / (np.cos(sol_altitude) * np.cos(latitude))))
    
    solar_azimuthy = 180 - bool_check*addition - (bool_check-1)*addition
    
    return solar_azimuthy


def get_sun_vector(sol_azimuth: float, sol_altitude: float):
    """
            Generates the sun vector to calculate the power on pv module.
            North = 0°, West = -90°, East = +90°, South = 180

            s = ( cos αs ∙ cos γs,
                  − sin αs ∙ cos γs,
                  sin γs
                )



                Args:
                    sol_altitude: solar altitude in degreees based on hour angle.
                    sol_azimuth: sun azimuth in degrees.

                Returns:
                    sun vector.
    """

    """
        Validation
    """
    validate_sol_altitude(sol_altitude)
    validate_sol_azimuth(sol_azimuth)
    """
        End Validation. Calculation starts.
    """

    """
    Convert to radians.
    """
    sol_altitude = np.radians(sol_altitude)
    sol_azimuth = np.radians(sol_azimuth)

    vecky = np.array([np.cos(sol_azimuth)*np.cos(sol_altitude),
                      -np.sin(sol_azimuth)*np.cos(sol_altitude),
                      np.sin(sol_altitude)])
    return vecky


def get_module_vector(module_azimuth: float, module_angle: typing.Union[int, float]):
    """
            Generates the module vector to calculate the power on a pv module.


            𝐧 = (
                    − cos αE ∙ sin β,
                    sin αE ∙ sin β,
                    cos β
                )




                Args:
                    module_angle: module angle to horizontal in degreees from 0 to 90.
                    module_azimuth: module azimuth in degrees North = 0°, West = -90°, East = +90°, South = 180.

                Returns:
                    module vector.
    """

    """
        Validation
    """
    validate_module_angle(module_angle)
    validate_module_azimuth(module_azimuth)
    """
        End Validation. Calculation starts.
    """

    """
    Convert to radians.
    """
    module_angle = np.radians(module_angle)
    module_azimuth = np.radians(module_azimuth)

    vecky = np.array([-np.cos(module_azimuth) * np.sin(module_angle),
                      np.sin(module_azimuth) * np.sin(module_angle),
                      np.cos(module_angle)])
    return vecky


def get_angle_incidence(sun_vectory: np.ndarray, module_vectory: np.ndarray):
    """
    calculates angle of incidence by sun and module vectors.
    θ = arccos(s*n)
    = arccos(− cos αs ∙ cos γs ∙ cos αE ∙ sin β − sin αs ∙ cos γs ∙ sin αE ∙ sin β + sin γs ∙ cos β)
    = arccos(− cos γs ∙ sin β ∙ cos(αs − αE) + sin γs ∙ cos β)

    Args:
            sun_vectory: The solar vector dependent on azimuth and altitude.
            module_vectory: The module vector dependent on azimuth and angle.

    Returns:
            angle of incidence in degrees.
    """
    if sun_vectory.shape[0] == 3:  # Transpose if necessary
        sun_vectory = sun_vectory.T
    if module_vectory.shape[0] == 3:
        module_vectory = module_vectory.T

    if sun_vectory.ndim == 1:
      sun_vectory = sun_vectory[None, :]  # Add a dimension if needed
    if module_vectory.ndim == 1:
      module_vectory = module_vectory[None, :]

    sun_vectors_norm = sun_vectory / np.linalg.norm(sun_vectory, axis=-1)[..., None]
    module_vectors_norm = module_vectory / np.linalg.norm(module_vectory, axis=-1)[..., None]



    # dot_products = np.einsum('ik,j->i', sun_vectors_norm, module_vectors_norm)
    # angles_degrees = np.degrees(np.arccos(dot_products))

    angles_degrees = np.empty(sun_vectory.shape[0])
        
    for i in range(len(sun_vectory)):
        dot_product = np.dot(sun_vectors_norm[i], module_vectors_norm[-1])
        angles_degrees[i] = np.degrees(np.arccos(dot_product))

    return angles_degrees


def get_direct_rad_module(direct_rad: float, angle_incidencey: float, sol_altitude: float):
    """
    EDirekt_Sol = EDirekt_H ∙ cos θ/sin γs
    Args:
            direct_rad: total radiation on the perpendicular to the sun in W/m².
            angle_incidencey: The angle of incidence in degrees.
            sol_altitude: the sun altitude in degrees.

    Returns:
            direct radiation on the module in W/m²
    """

    """
    Validation
    """
    validate_direct_radiation(direct_rad)
    validate_angle_incidence(angle_incidencey)
    validate_sol_altitude(sol_altitude)

    """
    Calculation
    """
    angle_incidencey = np.radians(angle_incidencey)
    sol_altitude = np.radians(sol_altitude)
    cosincy = np.cos(angle_incidencey)
    sinalt = np.sin(sol_altitude)
    factors = cosincy/sinalt
    factors = np.clip(factors, 0, 1)
    return np.multiply(direct_rad, factors)  # at night, this bitch turns negative. No sunlight is being stolen under my watch!


def get_diffuse_rad_module(diffuse_rad: float, module_angle: typing.Union[int, float]):
    """
    EDiffus_Sol = ½ ∙ EDiffus_H ∙ (1 + cos β)
    Args:
            diffuse_rad: diffuse radiation (W/m²).
            module_angle: The angle of the pv module in degrees between 0 and 90.

    Returns:
            diffus radiation on the module in W/m²
    """

    """
    Validation
    """
    validate_diffuse_radiation(diffuse_rad)
    validate_module_angle(module_angle)

    """
    Calculation
    """
    module_angle = np.radians(module_angle)

    return 0.5*diffuse_rad*(1+np.cos(module_angle))


def get_reflect_rad_module(glob_rad: float, module_angle: typing.Union[int, float], albedo_value: float = 0.2):
    """
    EDiffus_Sol = ½ ∙ EDiffus_H ∙ (1 + cos β)
    Args:
            glob_rad: global radiation = direct + diffus. (W/m²).
            module_angle: The angle of the pv module in degrees between 0 and 90.
            albedo_value: The albedo value, usually 0.2.

    Returns:
            diffus radiation on the module in W/m²
    """

    """
    Validation
    """
    validate_global_radiation(glob_rad)
    validate_module_angle(module_angle)
    validate_albedo_value(albedo_value)

    """
    Calculation
    """
    module_angle = np.radians(module_angle)

    return 0.5*glob_rad*(1-np.cos(module_angle))*albedo_value
