from main_pv.classes import *
import pandas as pd
import numpy as np
import datetime


"""
These values are often changed. So they can be changed directly here.
"""
"""
Building
"""
smart = False  # do the buildings posses smart meters?
battery_size_1 = 0  # the battery size of building 1 in kWh
battery_size_2 = 0  # the battery size of building 2 in kWh
"""
System
"""
grid_oriented = False  # do we operate grid oriented or pv oriented? if grid oriented, we simply spread the electric car demand.
"""
Heating circuits
"""
flow_temp_1 = 30  # degrees celsius for the flow temperature for house 1
ret_temp_1 = 25  # degrees celsius for the return temperature for house 1
flow_temp_2 = 30  # degrees celsius for the flow temperature for house 2
ret_temp_2 = 25  # degrees celsius for the return temperature for house 2
"""
PV
"""
pv_power_1 = 15  # Size of pv field for house 1 in kWp
pv_azimuth_1 = 180  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_1 = 35  # the angle above the horizon of the pv field for house 1.
pv_power_2_1 = 10  # Size of first pv field for house 2 in kWp
pv_azimuth_2_1 = 90  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_2_1 = 60  # the angle above the horizon of the first pv field for house 2.
pv_power_2_2 = 10  # Size of second pv field for house 2 in kWp
pv_azimuth_2_2 = -90  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_2_2 = 60  # the angle above the horizon of the second pv field for house 2.


"""
Doing the rest of the things.
"""
el_schedule_1 = [None for _ in range(7)]
time_go_1 = datetime.time(7, 20)
duration_1 = datetime.timedelta(hours=10, minutes= 10)
distance_1 = 40  # kilometer

for i in range(7):
    el_schedule_1[i] = ElCarCycle(time_go_1, duration_1, distance_1, i)

car_1_1 = ElCar(battery_cap=80, max_charge_pow= 8, efficiency= 3, el_schedule= el_schedule_1)  # we're safe around mom, right?
car_1_2 = ElCar(battery_cap=80, max_charge_pow= 8, efficiency= 3, el_schedule= el_schedule_1)  # we're safe around mom, right?

el_schedule_2 = [None for _ in range(7)]
time_go_2 = datetime.time(14, 20)
duration_2 = datetime.timedelta(hours=0, minutes= 20)
distance_2 = 8  # kilometer

for i in range(7):
    el_schedule_2[i] = ElCarCycle(time_go_2, duration_2, distance_2, i)

geta = ElCar(battery_cap=50, max_charge_pow= 11, efficiency= 5, el_schedule= el_schedule_2)  # we're safe around mom, right?
caracalla = ElCar(battery_cap=50, max_charge_pow=11, efficiency=5, el_schedule=el_schedule_2)  # He has returned!!! Geta, watch out!

cars_1 = [car_1_1, car_1_2]
roman_chariots = [geta, caracalla]

heater_1 = HeatingSystem(
    sys_type_ind=2,
    sub_sys_type_ind=5, 
    eng_source=0, 
    year=2020, 
    power=40, 
    cost=0, 
    share=0.6)  # air water heat pump. Electricity as eng source. Providing 60 % of the total energy to the circuit.
heater_2 = HeatingSystem(
    sys_type_ind=0, 
    sub_sys_type_ind=0, 
    eng_source=2, 
    year=1998, 
    power=100, 
    cost=0, 
    share=0.4)  # Low Temp Boiler. Oil as energy source. Providing 40 % of the total energy to the circuit.

heat_circuit_1 = HeatingCircuit(
    flow_temp=flow_temp_1, 
    ret_temp=ret_temp_1, 
    share=1, 
    heating_systems=[heater_1, heater_2])  # heating circuit for house 1. Only heating circuit, so it provides 100 % of the energy the house needs.

heater_3 = HeatingSystem(
    sys_type_ind=2,
    sub_sys_type_ind=5, 
    eng_source=0, 
    year=2020, 
    power=20, 
    cost=0, 
    share=0.8)  # air water heat pump. Electricity as eng source. Providing 80 % of the total energy to the circuit.
heater_4 = HeatingSystem(
    sys_type_ind=0, 
    sub_sys_type_ind=3, 
    eng_source=3, 
    year=2020, 
    power=30, 
    cost=0, 
    share=0.2)  # Condenser Boiler. Wood-Pellets as energy source. Providing 20 % of the total energy to the circuit.

heat_circuit_2 = HeatingCircuit(
    flow_temp=flow_temp_2, 
    ret_temp=ret_temp_2, 
    share=1, 
    heating_systems=[heater_3, heater_4])  # heating circuit for house 2. Only heating circuit, so it provides 100 % of the energy the house needs.

heat_circuits_1 = [heat_circuit_1]
heat_circuits_2 = [heat_circuit_2]

pv_1 = PVSystem(
    max_power=pv_power_1, 
    con_year=2020, 
    azimuth=pv_azimuth_1, 
    angle=pv_angle_1)
pv_2_1 = PVSystem(
    max_power=pv_power_2_1, 
    con_year=2020, 
    azimuth=pv_azimuth_2_1, 
    angle=pv_angle_2_1)
pv_2_2 = PVSystem(
    max_power=pv_power_2_2, 
    con_year=2020, 
    azimuth=pv_azimuth_2_2, 
    angle=pv_angle_2_2)

pv_group_1 = [pv_1]
pv_group_2 = [pv_2_1, pv_2_2]

home_1 = Building(
    h_a=None, 
    liv_area=200, 
    residents=5, 
    air_vol=None, 
    ht=None, 
    res_units=3, 
    con_year=1987, 
    large_el_devices=5, 
    dw_temp=60, 
    heating_circuits=heat_circuits_1, 
    e_cars=cars_1, 
    smartmeter=smart, 
    pv_sytem=pv_group_1, 
    base_el_demand=None,
    battery_cap=battery_size_1,
    roof_change=True,
    heat_change=True,
    is_in_bw=True,
    solar_thermal_area=4.4,
    have_renovation_roadmap=True)

# tari = get_pv_feed_in_tariff(8, True)
# print(tari)
# tari = get_pv_feed_in_tariff(25, False)
# print(tari)
# tari = get_pv_feed_in_tariff(800, True)
# print(tari)
# tari = get_pv_feed_in_tariff(8000, True)
# print(tari)
# tari = get_pv_feed_in_tariff(139, False)
# print(tari)

# res = get_amortization_time_pv(pv_cost=35000, yearly_el_self_use=11000, yearly_el_feed_in=27000, price_electricity=40, pv_max_power=35, opp_cost=0.05)
# print(res)
# pv_needed = home_1.get_pv_needed()
# print(pv_needed)

home_2 = Building(
    h_a=None, 
    liv_area=200, 
    residents=5, 
    air_vol=None, 
    ht=None, 
    res_units=3, 
    con_year=2020, 
    large_el_devices=5, 
    dw_temp=60, 
    heating_circuits=heat_circuits_2, 
    e_cars=roman_chariots, 
    smartmeter=smart, 
    pv_sytem=pv_group_2, 
    base_el_demand=None,
    battery_cap=battery_size_2)

en_manage_1 = EnergyManagement([home_1], grid_oriented=grid_oriented)
en_manage_2 = EnergyManagement([home_2], grid_oriented=grid_oriented)

en_manage_12 = EnergyManagement([home_1, home_2], grid_oriented=grid_oriented)

caracalla.initiate_eng_need_weekly_schedule()
car_schedule = caracalla.df_schedule

charge_schedule_stupid = en_manage_12.get_specific_weekly_stupid(caracalla)
charge_schedule_smart_no_pv = en_manage_12.get_specific_week_smartnopv(caracalla)

date_time_1 = np.datetime64(f"2024-01-01T13:50")
date_time_2 = np.datetime64(f"2024-01-01T20:20")
print(caracalla.is_in_garage(date_time_1))
print(caracalla.is_in_garage(date_time_2))
print(caracalla._get_energy_needed(date_time_1))
print(caracalla._get_energy_needed(date_time_2))
print(caracalla._get_charge_time_left(date_time_1))
print(caracalla._get_charge_time_left(date_time_2))
