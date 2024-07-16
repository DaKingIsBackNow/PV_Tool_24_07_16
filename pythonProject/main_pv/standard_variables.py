import os
import numpy as np

"""
General stuff
"""
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SOLPATH = "/databases/main/Solar4928-2023.csv"
MAINTYPES_PATH = "/databases/main/MainTypes.csv"
SUBTYPES_PATH = "/databases/main/SubTypes.csv"
ENGSRC_PATH = "/databases/main/EngSrcTypes.csv"
SUBSID_PATH = "/databases/main/SubsidyDesc.csv"
SUB_MAIN_PAIRINGS_PATH = "/databases/main/SubTypeNames.csv"
PCT_PATH = "/databases/seasons/PCTs.csv"
ELDEMAND_PATH = "/databases/main/ElDemand.csv"
DELIMITER = ";"

SUMMER_STRING = "Summer"
WINTER_STRING = "Winter"
SEASON_STRING = "Season"
AZIMUTH_STRING = "Azimuth"
ANGLE_STRING = "Angle"
HOUR_STRING = "Hour"
INDEX_NAMES= (SEASON_STRING, AZIMUTH_STRING, ANGLE_STRING, HOUR_STRING)
DIFF_STRING = "Diffuse_Power"
GLOB_STRING_ALT = "Global_Power"
DATETIME_STRING = "Datetime"
DAY_OF_YEAR_STRING = "DayOfYear"
TIME_STRING = "Time"
RAD_ON_MOD_STRING = "RadOnMod"
PERCENT_STRING = "PctOnModule"
LATITUDE = 48.78
LONGITUDE = 9.18
TIMEZONE = "Europe/Berlin"
ALBEDO = 0.2

GLOBAL_STRING = "ghi"
DIFF_HOR_STRING = "dhi"
DIFF_NORM_STRING = "dni"
TEMP_AIR_STRING = "temp_air"
WIND_SPEED_STRING = "wind_speed"

WEEK_INDNAMES = ["Weekday", "Time"]
YEAR_INDNAMES = ["Date", "Time"]
ELDEMAND_STRING = "ElDemand"
SUBTYPE_INDEX_STRING = "SubTypeIndex"
ENG_INDEX_STRING = "EngIndex"
HELP_ENG_STRING = "HelpEng"
SUBSINDEX_STRING = "SubsIndex"
MAIN_INDEX_STRING = "MainIndex"
EFF_STRING = "DefEfficiency"
EL_ENG_IND = 0
HEAT_PUMP_STRING = "Heat Pump"
TIMESTAMP_STRING = "Timestamp"
EL_CAR_ENERGY_NEEDED_STRING = "EnNeeded"

SELF_USE_STRING = "self_use"
FEED_IN_STRING = "feed_in"
FROM_GRID_STRING = "from_grid"

"""
Assumptions
"""
TIME_DIF = np.timedelta64(10, "m")
DEMAND_HOURS = 4000

DEFAULT_YEAR = 2023
FACTOR_LIV_AREA = 9
FACTOR_RESIDENTS = 200
FACTOR_BIG_EL_DEVICES = 200

DEFAULT_HEIGHT = 3  # m. Average height for an apartment room.
DEF_FAC_AIR_TOT_VOL = 1.3  # factor between the air volume and the total volume of the building.
DEFAULT_WW_TEMP = 60  # default warm water temperature.
DEFAULT_A_TO_VE = 2/3  # default hull area to total volume.
STANDARD_ROOM_TEMP = 20  # in ° Celsius
STANDARD_OUTSIDE_TEMP = -12  # in ° Celsius for a winter day. Needed to calculate the max demand.


DEF_MIN_FLOW_TEMP = 30  # °C for the minimum expected flow temperature.
MAX_HEAT_PUMP_FLOW_TEMP_FACTOR = 1.5  # heat pump is 50 percent more efficent at minimum flow temperatur of 30 in comparison to 70
HEAT_PUMP_OUT_TEMP_FACTORS = np.array([2.21/4, 2.39/4, 2.68/4, 2.91/4, 4/4, 4.9/4, 5.43/4, 6.85/4])  # as factors for the efficiency of the heat pump. 
# Taken from a real model of a Vitocal heat pump
HEAT_PUMP_OUT_TEMP_REF_TEMPS = np.array([-20, -15, -10, -7, 2, 7, 10, 20])  # the reference temperatures in the data sheet of the heat pump.
DEF_FLOW_TEMP = 70  # default flow temp in °C
DEF_RET_TEMP = 55  # default return temp in °C

AIR_DENSITY = 1000  # kg/m³
AIR_HEAT_CONSTANT = 1  # kJ/kgK
WARM_WATER_FACTOR = 100  # W/resident
MAX_TEMP_HEATING = 15  # °C. The maximum outside temperature, where the building still heats. Above this point there is only the warm water demand and help energy.
DEFAULT_EL_DEMAND = 1000  # kWh for electric demand not including heating or electric cars.

# https://animation.fer.hr/images/50036152/EUROCON%202021_Determining_Lithium-ion_Battery_One-way_Energy_Efficiencies_Influence_of_C-rate_and_Coulombic_Losses.pdf
DEFAULT_BATTERY_EFF_IN = 0.9  # the efficiency of charging the battery. from 0 to 1.
DEFAULT_BATTERY_EFF_OUT = 0.85  # the efficiency of discharging the battery. from 0 to 1.
DEFAULT_BATTERY_YEARLY_DEGRADATION = 0.98  # the yearly degradation of the battery. 1 means no degradation, 0.98 means 2 percent per year. from 0 to 1.
DEFAULT_BATTERY_AGE = 0  # default age of battery. Here brand new one.
DEFAULT_BATTERY_MAX_CHARGE = 0.5  # meaning a 10 kWh battery can load with 5 kW.
DEFAULT_BATTERY_MAX_DISCHARGE = 0.5  # meaning a 10 kWh battery can unload with 5 kW.

DAYS_TRANSITION_1 = 90
DAYS_TRANSITION_2 = 273

PV_STANDARD_METHOD_FACTOR = 0.6  # m² PV/m² Roof
PV_ADVANCED_METHOD_FACTOR = 0.75  # m² PV/m² Roof
PV_SIMPLE_METHOD_FACTOR = 0.06  # kWp/m² Built-up property area
PV_DEF_EFF_FACTOR = 0.182  # kWp/m²
SOLAR_THERMAL_FACTOR = 5.5  # m²/kWp

E_WAERM_G_FACTOR_NEED = 0.15
E_WAERM_G_PV_FACTOR = 0.02/E_WAERM_G_FACTOR_NEED  # the factor of how much the building needs for pv. in kWp/m² living area for 15%
E_WAERM_G_SOLTHERM_FACTOR_ONE_TWO_UNITS = 0.07/E_WAERM_G_FACTOR_NEED  # the factor of how much the building needs for solar thermal. in m²/m² living area for 15%. 
# For buildings with up to two residential units.
E_WAERM_G_SOLTHERM_FACTOR_MORE_UNITS = 0.06/E_WAERM_G_FACTOR_NEED  # the factor of how much the building needs for solar thermal. in m²/m² living area for 15%.
# for buildings with more than two residential units
E_WAERM_FACTOR_RENOVATION_ROADMAP = 0.05  # the factor in how much smaller the requirement for pv is if the building has a renovation roadmap. 
# the building needs 10 percent renewable equivalent instead of 15 currently.

"""
Typology values
"""
THRESHOLD_YEARS_AIR_EXCHANGE = np.array((1958, 1995, 2020))
VALUES_AIR_EXCHANGE = np.array((1, 0.8, 0.7, 0.6))

THRESHOLD_YEARS_HEAT_LOSS = np.array([1919, 1949, 1958, 1969, 1979, 1984, 1995])  # from typology tables for buildings in enev 2015
THRESHOLD_YEARS_HEAT_LOSS = np.append(THRESHOLD_YEARS_HEAT_LOSS, (2007, 2020))  # enev 2007 assumption and GEG 2020 assumption

VALUES_HEAT_LOSS = np.array([1.8, 1.4, 1.2, 1.1, 1, 0.9, 0.75, 0.7, 0.6, 0.4])  # from typology tables for buildings in enev 2015

"""
PV Module
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
# https://static.viessmann.com/resources/technical_documents/DE/de/VDP/4484027VDP00001_1.pdf?#pagemode=bookmarks&zoom=page-fit&view=Fit 
inv_pdc0 = 20000
eta_inv_nom=0.975
eta_inv_ref=0.968


"""
PV Feed in Tariffs. Part feed in.
"""
PV_THRESHOLDS_PART_FEED_IN = np.array([10, 40, 1000])  # kWp
PV_TARIFFS_PART_FEED_IN = np.array([82, 71, 58])  # Euro/MWh
"""
PV Feed in Tariffs. Full feed in.
"""
PV_THRESHOLDS_FULL_FEED_IN = np.array([10, 100, 400, 1000])  # kWp
PV_TARIFFS_FULL_FEED_IN = np.array([130, 109, 90, 77])  # Euro/MWh
