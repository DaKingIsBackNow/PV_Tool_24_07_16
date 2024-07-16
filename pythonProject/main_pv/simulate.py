import datetime
import math
import typing
import pandas as pd
import main_pv.solar_functions as solar_functions

from main_pv.classes import Building

"""
Stuttgart
"""
LATITUDE = 48.77
LONGITUDE = 9.18


def deg_to_rad(deg):
    """
    Converts degrees to radians
    """
    return deg / 180 * math.pi


def deg_func(math_func: typing.Callable, degrees: typing.Union[int, float]) -> float:
    """
    Apply a mathematical function to an angle given in degrees.

    Args:
    - math_func (callable): The mathematical function to apply.
    - degrees (int or float): The angle in degrees.

    Returns:
    - float: The result of applying the function to the angle in radians.
    """
    try:
        radians = degrees / 180 * math.pi
        return math_func(radians)
    except Exception as e:
        print(f"Error: {e}")
        return float('nan')  # Return NaN for invalid inputs or errors


def sim(homie: Building):
    schedule = pd.read_csv("Databases/Main/Schedule.csv", delimiter=";", index_col=0)
    el_demand_year = homie.liv_area * 9 + 200 * (homie.residents + homie.large_el_devices)  # kWh
    f = el_demand_year / 1000  # factor to the base values
    e_help = 0
    for heating in homie.heating_circuits:
        e_help += heating.help_electricity  # kW
    pass


def create_schedule(homie: Building):
    path = "Databases/Main/Schedule.csv"
    schedule = pd.read_csv(path, delimiter=";", index_col=0)
    tim = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    delta = datetime.timedelta(minutes=10)
    while tim.year < 2025:
        schedule.loc[tim] = 1
        tim = tim + delta


def create_el_car_demand(homie: Building):
    """
    all houses come here and are sent to the sub functions depending on if the house has a smart meter or not
    :param homie: The house object that should be analyzed
    """
    if not homie.smartmeter:
        dumb_el_car_demand(homie)
    else:
        smart_el_car_demand(homie)


def dumb_el_car_demand(homie: Building):
    schedule = pd.DataFrame()
    tim = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    delta = datetime.timedelta(minutes=10)
    zeroes = False
    if homie.e_cars is None:
        zeroes = True
    while tim.year < 2025:
        if not zeroes:
            power = 0
            for e_car in homie.e_cars:
                schedule.loc[tim] = 0
        else:
            schedule.loc[tim] = 0
        tim = tim + delta


def smart_el_car_demand(homie: Building):
    pass


def get_heatpump_demand():
    pass
