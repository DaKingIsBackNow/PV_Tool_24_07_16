from main_pv.classes import *
from main_pv.standard_variables import *
import pandas as pd
import numpy as np
import datetime


def get_ordinal_suffix(i):
    """Returns the ordinal suffix for a number (e.g., 'st', 'nd', 'rd', 'th')."""
    if 11 <= (i % 100) <= 13:
        return 'th'
    else:
        return {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')
    

def save_yearly_scenarios(en_manage_1: EnergyManagement, en_manage_2: EnergyManagement, en_manage_12: EnergyManagement, bat_caps):
    """
    Saving stuff to the csv.
    """
    data = []

    for smart_meter in [False, True]:
        for battery_cap in [np.zeros(len(bat_caps)), bat_caps]:  # Handle no battery and battery cases
            for grid_oriented in ([False] if not smart_meter else [False, True]):  # Grid orientation only for smart meters
                home_1.smartmeter = smart_meter
                home_2.smartmeter = smart_meter
                home_1.battery_cap = battery_cap[0]
                home_2.battery_cap = battery_cap[1]
                en_manage_1.reset_vars(grid_oriented=grid_oriented)
                en_manage_2.reset_vars(grid_oriented=grid_oriented)
                en_manage_12.reset_vars(grid_oriented=grid_oriented)
                # Calculate the results for each building and the system.
                results_1_df = en_manage_1.simulate_year()
                results_2_df = en_manage_2.simulate_year()
                results_12_df = en_manage_12.simulate_year()
                # index to datetime.
                results_1_df.index = pd.to_datetime(results_1_df.index)
                results_2_df.index = pd.to_datetime(results_2_df.index)
                results_12_df.index = pd.to_datetime(results_12_df.index)
                # Calculate sums. Divide by six to get kWh because of ten minute intervals.
                vals_1 = results_1_df.sum()/6
                vals_2 = results_2_df.sum()/6
                vals_12 = results_12_df.sum()/6

                data.append({
                        "Smart Meter": smart_meter,
                        "Battery": battery_cap,
                        "Grid Oriented": grid_oriented,
                        "Building 1 Self Use (kWh)": vals_1.loc[SELF_USE_STRING],
                        "Building 1 Feed-in (kWh)": vals_1.loc[FEED_IN_STRING],
                        "Building 1 From Grid (kWh)": vals_1.loc[FROM_GRID_STRING],
                        "Building 2 Self Use (kWh)": vals_2.loc[SELF_USE_STRING],
                        "Building 2 Feed-in (kWh)": vals_2.loc[FEED_IN_STRING],
                        "Building 2 From Grid (kWh)": vals_2.loc[FROM_GRID_STRING],
                        "System Self Use (kWh)": vals_12.loc[SELF_USE_STRING],
                        "System Feed-in (kWh)": vals_12.loc[FEED_IN_STRING],
                        "System From Grid (kWh)": vals_12.loc[FROM_GRID_STRING],
                    })

    df = pd.DataFrame(data)
    df.to_csv("simulation_results.csv", index=False, sep=DELIMITER, decimal=",")


def save_monthly_scenarios(en_manage_1: EnergyManagement, en_manage_2: EnergyManagement, en_manage_12: EnergyManagement, bat_caps):
    """
    Saving stuff to the csv.
    """
    data = []

    for smart_meter in [False, True]:
        for battery_cap in [np.zeros(len(bat_caps)), bat_caps]:  # Handle no battery and battery cases
            for grid_oriented in ([False] if not smart_meter else [False, True]):  # Grid orientation only for smart meters
                home_1.smartmeter = smart_meter
                home_2.smartmeter = smart_meter
                home_1.battery_cap = battery_cap[0]
                home_2.battery_cap = battery_cap[1]
                en_manage_1.reset_vars(grid_oriented=grid_oriented)
                en_manage_2.reset_vars(grid_oriented=grid_oriented)
                en_manage_12.reset_vars(grid_oriented=grid_oriented)
                # Calculate the results for each building and the system.
                results_1_df = en_manage_1.simulate_year()
                results_2_df = en_manage_2.simulate_year()
                results_12_df = en_manage_12.simulate_year()
                # index to datetime.
                results_1_df.index = pd.to_datetime(results_1_df.index)
                results_2_df.index = pd.to_datetime(results_2_df.index)
                results_12_df.index = pd.to_datetime(results_12_df.index)
                # Resample to monthly and calculate sums
                monthly_1 = results_1_df.resample("M").sum()/6
                monthly_2 = results_2_df.resample("M").sum()/6
                monthly_12 = results_12_df.resample("M").sum()/6

                # Iterate through each month and store data
                for month in monthly_1.index:
                    data.append({
                        "Smart Meter": smart_meter,
                        "Battery": battery_cap,
                        "Grid Oriented": grid_oriented,
                        "Month": month.month,
                        "Building 1 Self Use (kWh)": monthly_1.loc[month, SELF_USE_STRING],
                        "Building 1 Feed-in (kWh)": monthly_1.loc[month, FEED_IN_STRING],
                        "Building 1 From Grid (kWh)": monthly_1.loc[month, FROM_GRID_STRING],
                        "Building 2 Self Use (kWh)": monthly_2.loc[month, SELF_USE_STRING],
                        "Building 2 Feed-in (kWh)": monthly_2.loc[month, FEED_IN_STRING],
                        "Building 2 From Grid (kWh)": monthly_2.loc[month, FROM_GRID_STRING],
                        "System Self Use (kWh)": monthly_12.loc[month, SELF_USE_STRING],
                        "System Feed-in (kWh)": monthly_12.loc[month, FEED_IN_STRING],
                        "System From Grid (kWh)": monthly_12.loc[month, FROM_GRID_STRING],
                    })

    df = pd.DataFrame(data)
    df.to_csv("simulation_results_monthly.csv", index=False, sep=DELIMITER, decimal=",")
    

def get_pvs_needed(buildings: list[Building], solar_thermal_planned: list[float] = None, print_at_end= True):
    """
    Gives back an array of the kWp needed for a list of buildings.
    """
    planned_solars = np.zeros(len(buildings)) if solar_thermal_planned is None else np.array(solar_thermal_planned)
    if len(planned_solars) != len(buildings):
        raise ValueError("The lists for solar thermal thingies and buildings needs to be the same length.")
    pv_needed = np.empty(shape=len(buildings))
    for i, building in enumerate(buildings):
        pv_needed[i] = building.get_pv_needed(planned_solars[i])
    if print_at_end:
        print_pv_needed(pv_needed)
    return pv_needed


def print_pv_needed(pv_needed: np.ndarray):
    """
    Prints an array of the kWp needed for an array of pvs and buildings.
    """
    for i, pv_n in enumerate(pv_needed):
        if pv_n == 0:
            print("You actually don't need anything. Consider youurself lucky!")
            continue
        suffix = get_ordinal_suffix(i+1)
        print(f"The {i+1}{suffix} Building needs {round(pv_n, 2)} kWp legally. Better get on it buddy.")
    return


def print_results(en_manage_1: EnergyManagement, en_manage_2: EnergyManagement, en_manage_12: EnergyManagement):
    """
    Prints the results of the current scenario and all that. It also calculates them, I guess.
    """
    results_1_df = en_manage_1.simulate_year()
    results_2_df = en_manage_2.simulate_year()
    results_12_df = en_manage_12.simulate_year()
    # index to datetime.
    results_1_df.index = pd.to_datetime(results_1_df.index)
    results_2_df.index = pd.to_datetime(results_2_df.index)
    results_12_df.index = pd.to_datetime(results_12_df.index)
    # Calculate sums. Divide by six to get kWh because of ten minute intervals.
    vals_1 = results_1_df.sum()/6
    vals_2 = results_2_df.sum()/6
    vals_12 = results_12_df.sum()/6

    smart_meter = [en_manage_1.are_all_smart(), en_manage_2.are_all_smart(), en_manage_12.are_all_smart()]
    battery_cap = [en_manage_1.get_battery_cap(), en_manage_2.get_battery_cap(), en_manage_12.get_battery_cap()]
    griddey_oriented = [en_manage_1.grid_oriented, en_manage_2.grid_oriented, en_manage_12.grid_oriented]

    print(f'Smart Meter: {smart_meter}, \
            \n Battery: {battery_cap},\
            \n Grid Oriented: {griddey_oriented},\
            \n Building 1 Self Use (kWh): {vals_1.loc[SELF_USE_STRING]},\
            \n Building 1 Feed-in (kWh): {vals_1.loc[FEED_IN_STRING]},\
            \n Building 1 From Grid (kWh): {vals_1.loc[FROM_GRID_STRING]},\
            \n Building 2 Self Use (kWh): {vals_2.loc[SELF_USE_STRING]},\
            \n Building 2 Feed-in (kWh): {vals_2.loc[FEED_IN_STRING]},\
            \n Building 2 From Grid (kWh): {vals_2.loc[FROM_GRID_STRING]},\
            \n System Self Use (kWh): {vals_12.loc[SELF_USE_STRING]},\
            \n System Feed-in (kWh): {vals_12.loc[FEED_IN_STRING]},\
            \n System From Grid (kWh): {vals_12.loc[FROM_GRID_STRING]},\
        ')



"""
These values are often changed. So they can be changed directly here.
"""
"""
Building
"""
smarts = (True, True)  # do the buildings possess smart meters? First value for building 1 and second value for building 2.
battery_sizes = (10, 40)  # the battery sizes of the buildings in kWh
living_areas = (200, 200)  # the battery sizes of the buildings in kWh
construction_years = (1987, 2020)  # the battery sizes of the buildings in kWh
"""
System
"""
grid_oriented = (False, False, False)  # do we operate grid oriented or pv oriented? 
# if grid oriented, we simply spread the electric car demand.
# First value is for building 1, second for building 2 and third for the combined system.
"""
Heating circuits
"""
flow_temps = (30, 30)  # degrees celsius for the flow temperatures
ret_temps = (25, 25)  # degrees celsius for the return temperatures

"""
PV
"""
"""
Building 1
"""
pv_power_1 = 15  # Size of pv field for house 1 in kWp
pv_azimuth_1 = 180  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_1 = 35  # the angle above the horizon of the pv field for house 1.
"""
Building 2
"""
pv_power_2_1 = 10  # Size of first pv field for house 2 in kWp
pv_azimuth_2_1 = 90  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_2_1 = 60  # the angle above the horizon of the first pv field for house 2.
pv_power_2_2 = 10  # Size of second pv field for house 2 in kWp
pv_azimuth_2_2 = -90  # module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
pv_angle_2_2 = 60  # the angle above the horizon of the second pv field for house 2.

"""
Actual initialization for the variables and systems.
"""
"""
Electric cars. Home 1.
"""
el_schedule_1 = [None for _ in range(7)]
time_go_1 = datetime.time(7, 20)
duration_1 = datetime.timedelta(hours=10, minutes= 10)
distance_1 = 40  # kilometer

for i in range(7):
    el_schedule_1[i] = ElCarCycle(time_go_1, duration_1, distance_1, i)

car_1_1 = ElCar(battery_cap=50, max_charge_pow= 8, efficiency= 4, el_schedule= el_schedule_1)  # we're safe around mom, right?
car_1_2 = ElCar(battery_cap=50, max_charge_pow= 8, efficiency= 4, el_schedule= el_schedule_1)  # we're safe around mom, right?

"""
Electric cars. Home 2.
"""
el_schedule_2 = [None for _ in range(7)]
time_go_2 = datetime.time(14, 20)
duration_2 = datetime.timedelta(hours=0, minutes= 20)
distance_2 = 5  # kilometer

for i in range(7):
    el_schedule_2[i] = ElCarCycle(time_go_2, duration_2, distance_2, i)

geta = ElCar(battery_cap=80, max_charge_pow= 11, efficiency= 5, el_schedule= el_schedule_2)  # we're safe around mom, right?
caracalla = ElCar(battery_cap=80, max_charge_pow=11, efficiency=5, el_schedule=el_schedule_2)  # He has returned!!! Geta, watch out!

cars_1 = [car_1_1, car_1_2]
roman_chariots = [geta, caracalla]

"""
Heating circuit and heaters for home 1.
"""
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
    flow_temp=flow_temps[0], 
    ret_temp=ret_temps[0], 
    share=1, 
    heating_systems=[heater_1, heater_2])  # heating circuit for house 1. Only heating circuit, so it provides 100 % of the energy the house needs.

"""
Heating circuit and heaters for home 2.
"""
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
    flow_temp=flow_temps[1], 
    ret_temp=ret_temps[1], 
    share=1, 
    heating_systems=[heater_3, heater_4])  # heating circuit for house 2. Only heating circuit, so it provides 100 % of the energy the house needs.

heat_circuits_1 = [heat_circuit_1]
heat_circuits_2 = [heat_circuit_2]

"""
PV House 1.
"""
pv_1 = PVSystem(
    max_power=pv_power_1, 
    con_year=2020, 
    azimuth=pv_azimuth_1, 
    angle=pv_angle_1)
"""
PV House 2.
"""
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

"""
Roof Stuff. Does not matter for the calculation. Only matters to calculate the amount of pv needed if roof will be insulated soon.
"""

"""
Roof onbstructions house 1
"""
roof_obstr_1_1 = RoofObstruction(5, "Big Window")
roof_obstr_1_2 = RoofObstruction(2, "Chimney")
roof_obstruct_group_1 = [roof_obstr_1_1, roof_obstr_1_2]
"""
Roof obstructions house 2
"""
roof_obstr_2_1 = RoofObstruction(4, "Big Window")
roof_obstr_2_2 = RoofObstruction(1, "Chimney")
roof_obstruct_group_2 = [roof_obstr_2_1, roof_obstr_2_2]
"""
Roofs house 1
"""
roof_1_1 = Roof(real_area=80, azimuth=180, angle=35, roof_objects=roof_obstruct_group_1, is_main_roof=True, total_area=80)
roof_1_2 = Roof(real_area=80, azimuth=0, angle=35, roof_objects=None, is_main_roof=True, total_area=80)
roofs_house_1 = [roof_1_1, roof_1_2]
"""
Roofs house 2 Needed to calculate legal PV needed.
"""
roof_2_1 = Roof(real_area=80, azimuth=90, angle=60, roof_objects=roof_obstruct_group_2, is_main_roof=True, total_area=130)  # main roof east
roof_2_1_1 = Roof(real_area=25, azimuth=90, angle=10, roof_objects=None, is_main_roof=False, total_area=None)  # dormer east
roof_2_2 = Roof(real_area=80, azimuth=-90, angle=60, roof_objects=None, is_main_roof=True, total_area=130)  # main roof west
roof_2_2_1 = Roof(real_area=25, azimuth=-90, angle=10, roof_objects=None, is_main_roof=False, total_area=None)  # dormer west
roofs_house_2 = [roof_2_1, roof_2_1_1, roof_2_2, roof_2_2_1]

"""
Changes planned or recently made. Needed to calculate legal PV needed.
"""
heater_change_house_1 = True
heater_change_house_2 = False

roof_change_house_1 = False
roof_change_house_2 = True
"""
If official renovation plans exist. Needed to calculate legal PV needed.
"""
ren_plan_exists_house_1 = False
ren_plan_exists_house_2 = False
"""
Amount of solar thermal panels that exist for the houses already. Needed to calculate legal PV needed.
"""
solar_thermal_1 = 0
solar_thermal_2 = 0
"""
Amount of planned solar thermal panels that the houses want to install. Needed to calculate legal PV needed.
"""
planned_solar_thermal_1 = 0
planned_solar_thermal_2 = 8.8
"""
main_pv
"""
home_1 = Building(
    h_a=None, 
    liv_area=living_areas[0], 
    residents=5, 
    air_vol=None, 
    ht=None, 
    res_units=3, 
    con_year=construction_years[0], 
    large_el_devices=5, 
    dw_temp=60, 
    heating_circuits=heat_circuits_1, 
    e_cars=cars_1, 
    smartmeter=smarts[0], 
    pv_sytem=pv_group_1, 
    base_el_demand=None,
    battery_cap=battery_sizes[0],
    roofs=roofs_house_1,
    is_in_bw=True,
    heat_change=heater_change_house_1,
    roof_change=roof_change_house_1,
    have_renovation_roadmap=ren_plan_exists_house_1,
    solar_thermal_area=solar_thermal_1)

home_2 = Building(
    h_a=None, 
    liv_area=living_areas[1], 
    residents=5, 
    air_vol=None, 
    ht=None, 
    res_units=3, 
    con_year=construction_years[1], 
    large_el_devices=5, 
    dw_temp=60, 
    heating_circuits=heat_circuits_2, 
    e_cars=roman_chariots, 
    smartmeter=smarts[1], 
    pv_sytem=pv_group_2, 
    base_el_demand=None,
    battery_cap=battery_sizes[1],
    roofs=roofs_house_2,
    is_in_bw=True,
    heat_change=heater_change_house_2,
    roof_change=roof_change_house_2,
    have_renovation_roadmap=ren_plan_exists_house_2,
    solar_thermal_area=solar_thermal_2)

"""
Energy management systems. Once for main_pv separately and once as one system.
"""
energy_manage_1 = EnergyManagement([home_1], grid_oriented=grid_oriented[0])
energy_manage_2 = EnergyManagement([home_2], grid_oriented=grid_oriented[1])

energy_manage_12 = EnergyManagement([home_1, home_2], grid_oriented=grid_oriented[2])


"""
Calculations and results.
"""
"""
Printing the current situation if you want.
"""
print_results(energy_manage_1, energy_manage_2, energy_manage_12)
"""
Defining batteries for scenarios where there are batteries.
"""
battery_sizes = (0, 50)
"""
Sending the data to be simulated and returned for the scenarios.
"""
save_monthly_scenarios(energy_manage_1, energy_manage_2, energy_manage_12, battery_sizes)
save_yearly_scenarios(energy_manage_1, energy_manage_2, energy_manage_12, battery_sizes)

"""
Doing the pv_needed stuff.
"""
get_pvs_needed([home_1, home_2], [planned_solar_thermal_1, planned_solar_thermal_2])
get_num_modules(4, 0.33)


"""
A bunch of old testing stuff.
"""

# """
    # Results for no smart meter and no battery. (Not grid oriented because impossible without smart meters)
    # """
    # print("\nThese are the results for NO Smart meter and NO Battery:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = False
    # home_2.smartmeter = False
    # home_1.battery_cap = 0
    # home_2.battery_cap = 0
    # grid_oriented = False
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)
    # """
    # calculate
    # """
    # print_results(en_manage_1, en_manage_2, en_manage_12)
    # """
    # Results for smart meter but no battery. grid oriented.
    # """
    # print("\nThese are the results for YES smart meter but NO battery. YES Grid oriented:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = True
    # home_2.smartmeter = True
    # home_1.battery_cap = 0
    # home_2.battery_cap = 0
    # grid_oriented = True
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)
    
    # print_results(en_manage_1, en_manage_2, en_manage_12)
    # """
    # Results for smart meter but no battery. Not grid oriented.
    # """
    # print("\nThese are the results for YES Smart meter but NO Battery. NOT Grid oriented:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = True
    # home_2.smartmeter = True
    # home_1.battery_cap = 0
    # home_2.battery_cap = 0
    # grid_oriented = False
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)

    # print_results(en_manage_1, en_manage_2, en_manage_12)

    # """
    # Results for no smart meter but battery. (Not grid oriented because impossible without smart meters)
    # """
    # print("\nThese are the results for NO Smart meter but YES Battery:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = False
    # home_2.smartmeter = False
    # home_1.battery_cap = battery_size_1
    # home_2.battery_cap = battery_size_2
    # grid_oriented = False
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)
    
    # print_results(en_manage_1, en_manage_2, en_manage_12)
    # """
    # Results for smart meter and battery. grid oriented.
    # """
    # print("\nThese are the results for YES smart meter and YES battery. YES Grid oriented:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = True
    # home_2.smartmeter = True
    # home_1.battery_cap = battery_size_1
    # home_2.battery_cap = battery_size_2
    # grid_oriented = True
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)

    # print_results(en_manage_1, en_manage_2, en_manage_12)

    # """
    # Results for smart meter and battery. Not grid oriented.
    # """
    # print("\nThese are the results for YES smart meter and YES battery. NOT Grid oriented:\n")
    # """
    # Initial setup
    # """
    # home_1.smartmeter = True
    # home_2.smartmeter = True
    # home_1.battery_cap = battery_size_1
    # home_2.battery_cap = battery_size_2
    # grid_oriented = False
    # en_manage_1.reset_vars(grid_oriented=grid_oriented)
    # en_manage_2.reset_vars(grid_oriented=grid_oriented)
    # en_manage_12.reset_vars(grid_oriented=grid_oriented)

    # print_results(en_manage_1, en_manage_2, en_manage_12)
    # return
# def print_results(en_manage_1: EnergyManagement, en_manage_2: EnergyManagement, en_manage_12: EnergyManagement):
#     """
#     Prints stuff.
#     """
#     """
#     calculate
#     """
#     results_1 = en_manage_1.simulate_year()
#     results_1 = str(f"self use amount was {round(results_1[0], 2)} kWh, the feed in amount was "
#                       f"{round(results_1[1], 2)} kWh, and the amount needed from the grid was {round(results_1[2], 2)} kWh")
#     results_2 = en_manage_2.simulate_year()
#     results_2 = str(f"self use amount was {round(results_2[0], 2)} kWh, the feed in amount was "
#                     f"{round(results_2[1], 2)} kWh, and the amount needed from the grid was {round(results_2[2], 2)} kWh")
#     results_12 = en_manage_12.simulate_year()
#     results_12 = str(f"self use amount was {round(results_12[0], 2)} kWh, the feed in amount was "
#                     f"{round(results_12[1], 2)} kWh, and the amount needed from the grid was {round(results_12[2], 2)} kWh")
#     """
#     Print0
#     """
#     print(f"the results for building 1 on its own {results_1}")
#     print(f"the results for building 2 on its own {results_2}")
#     print(f"the results for the system of both buildings is {results_12}")


# def run_simulations_scenarios(en_manage_1: EnergyManagement, en_manage_2: EnergyManagement, en_manage_12: EnergyManagement, bat_caps: list[float]):
#     """
#     Runs simulations for 2 buildings. 
#     en_manage_1 is the energy management system of building 1, 
#     en_manage_2 is the energy management system of building 2,
#     en_manage_12 is the energy management system of both buildings working together like real hippies should.
#     bat_caps is a list of the battery sizes for the buildings in the cases of YES battery. It should have the size of two since there are two buildings.
#     """
#     data = []

#     for smart_meter in [False, True]:
#         for battery_cap in [np.zeros(len(bat_caps)), bat_caps]:  # Handle no battery and battery cases
#             for grid_oriented in ([False] if not smart_meter else [False, True]):  # Grid orientation only for smart meters
#                 home_1.smartmeter = home_2.smartmeter = smart_meter
#                 battery_size_1, battery_size_2 = battery_cap
#                 home_1.battery_cap = battery_size_1
#                 home_2.battery_cap = battery_size_2
#                 # Reset the energy management objects
#                 en_manage_1.reset_vars(grid_oriented=grid_oriented)
#                 en_manage_2.reset_vars(grid_oriented=grid_oriented)
#                 en_manage_12.reset_vars(grid_oriented=grid_oriented)

#                 sm_string = "YES" if smart_meter else "NO"
#                 batt_string = "YES" if battery_cap[0] or battery_cap[1] else "NO"
#                 grid_string = "YES" if grid_oriented else "NO"

#                 print(f"\nSmart Meter: {sm_string}, Battery: {batt_string}, Grid Oriented: {grid_string}")
#                 print_results(en_manage_1, en_manage_2, en_manage_12)

#                 results_1 = en_manage_1.simulate_year()
#                 results_2 = en_manage_2.simulate_year()
#                 results_12 = en_manage_12.simulate_year()


#                 data.append({
#                     "Smart Meter": smart_meter,
#                     "Battery": battery_cap,
#                     "Grid Oriented": grid_oriented,
#                     "Building 1 Self Use (kWh)": results_1[0],
#                     "Building 1 Feed-in (kWh)": results_1[1],
#                     "Building 1 From Grid (kWh)": results_1[2],
#                     "Building 2 Self Use (kWh)": results_2[0],
#                     "Building 2 Feed-in (kWh)": results_2[1],
#                     "Building 2 From Grid (kWh)": results_2[2],
#                     "System Self Use (kWh)": results_12[0],
#                     "System Feed-in (kWh)": results_12[1],
#                     "System From Grid (kWh)": results_12[2]
#                 })

#     df = pd.DataFrame(data)

#     # Save to CSV
#     df.to_csv("simulation_results.csv", index=False, sep=DELIMITER, decimal=",")

# en_manage_12._get_db_week_simsmart_season(is_summer=True)

# print(en_manage.get_electricity_share_heating())

# weekly_thing = EnergyManagement.get_specific_week_smartnopv(caracalla)
# print(weekly_thing)
# yearly_stupid = en_manage._get_yearly_stupid(2024)

# path = DIR_PATH + "/Databases/Helper/test_year_stupid.csv"
# yearly_stupid.to_csv(path, sep= DELIMITER)

# def _categorize_years(year, thresholds, values):
#         """Categorizes years into bins based on thresholds and returns corresponding values."""
        
#         # Ensure NumPy arrays
#         thresholds = np.asarray(thresholds)
#         values = np.asarray(values)

#         # Use searchsorted to find the index of the category the year belongs to
#         category_index = np.searchsorted(thresholds, year, side='right') 

#         # Return the corresponding value (handle out-of-bounds years)
#         if 0 <= category_index < len(values):
#             return values[category_index]
#         else:
#             return None  # Or a default value for years outside the thresholds
        

# years_threshholds = np.array([1919, 1949, 1958, 1969, 1979, 1984, 1995])  # from typology tables for buildings in enev 2015
# years_threshholds = np.append(years_threshholds, (2007, 2020))  # enev 2007 assumption and GEG 2020 assumption

# values = np.array([1.8, 1.4, 1.2, 1.1, 1, 0.9, 0.75, 0.7, 0.6, 0.4])
# value = _categorize_years(4000, years_threshholds, values)

# print(value)

# import datetime
# import numpy as np
# import math
# import pvlib
# import os

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# SOLPATH = DIR_PATH + "/Databases/Main/Solar4928-2023.csv"
# DELIMITER = ";"
#
# SEASON_STRING = "Season"
# AZIMUTH_STRING = "Azimuth"
# ANGLE_STRING = "Angle"
# HOUR_STRING = "Hour"
# INDEX_NAMES= (SEASON_STRING, AZIMUTH_STRING, ANGLE_STRING, HOUR_STRING)
# DIFF_STRING = "Diffuse_Power"
# GLOB_STRING_ALT = "Global_Power"
# DATETIME_STRING = "Datetime"
# DAY_OF_YEAR_STRING = "DayOfYear"
# TIME_STRING = "Time"
# RAD_ON_MOD_STRING = "RadOnMod"
# PERCENT_STRING = "PctOnModule"
# LATITUDE = 48.78
# LONGITUDE = 9.18
# TIMEZONE = "Europe/Berlin"
# ALBEDO = 0.2
#
# GLOBAL_STRING = "ghi"
# DIFF_HOR_STRING = "dhi"
# DIFF_NORM_STRING = "dni"
# TEMP_AIR_STRING = "temp_air"
# WIND_SPEED_STRING = "wind_speed"

# datey_time = datetime.datetime(2024, 1, 1)
# print((datey_time - datetime.datetime(datey_time.year, 1, 1)).days)
# def try_it():
#     return True, "But why tho?", 45, 12


# tt = try_it()
# print(type(tt))
# print(tt)
# for i in range(len(tt)):
#     print(tt[i])


# tim = datetime.time(hour=20, minute=30)
# delly = datetime.timedelta(hours=5)
# suzy = (datetime.datetime.combine(datetime.date(year=2021, month=1, day=5), tim) + delly)
# print(tim, delly, suzy.time(), suzy.time() < tim)

# tim = datetime.datetime(year=2024, month=1, day=1)
# delta = datetime.timedelta(hours=1)
# counter = 0
# while tim.year < 2025:
#     counter += 1
#     if tim == tim + delta:
#         print("Eh?")
#     tim = tim + delta
#
# print(counter)


# lister = [0, 212, "jaso", 12.45, True]
#
# for lisy in lister:
#     print(lisy)

# arie = np.arange(start=0, stop=15, step=1)/5

# arie = 2*arie +0.5

# if type(arie) == np.ndarray:
#     if np.all(0 < arie) and np.all(arie < 3):
#         print("all smaller three bigger zero")
#     if np.any(np.logical_and(0 < arie, arie < 3)):
#         print("any smaller three bigger zero")

# cosi = np.cos(arie)

# print(cosi, arie)


# arie = np.array((0, 0.12, 0.4, 0.93, 12, 0.3, 99))


# arie = 0.6
# booly = np.float16(arie < 0.5)
# new_arie = arie - 0.3*booly - 100*(booly-1) 
# arie_1 = np.array((0, 1, 2))
# arie_2 = np.array((4, 12, 3))
# arie_3 = np.array((19, 14, 21))
# arie_4 = np.array((40, 11, 32))

# arie_13 = np.array([arie_1, arie_3])
# arie_24 = np.array([arie_2, arie_4])

# mul1 = np.matmul(arie_1, arie_2)
# mul2 = np.matmul(arie_3, arie_4)
# mul3 = np.diagonal(np.matmul(arie_13, arie_24.T))

# print("arie_13 shape:", arie_13.shape)
# print("arie_13:\n", arie_13) 

# arie_13 = arie_13[None, :] 
# print("arie_13 shape (after reshape):", arie_13.shape)
# print("arie_13:\n", arie_13)  

# sun_vectors_norm = arie_13 / np.linalg.norm(arie_13, axis=-1)[..., None]
# print("sun_vectors_norm shape:", sun_vectors_norm.shape)
# print("sun_vectors_norm:\n", sun_vectors_norm) 

# # arie_13 = arie_13[None, :]
# arie_4 = arie_4[None, :]

# # sun_vectors_norm = arie_13 / np.linalg.norm(arie_13, axis=1)[..., None]
# module_vector_norm = arie_4 / np.linalg.norm(arie_4, axis=-1)[..., None]
 
# print(sun_vectors_norm, module_vector_norm)

# print(np.arange(0, 91, 10))

# import pandas as pd
# import altair as alt
# import numpy as np
# import matplotlib.pyplot as plt

# # Read the data file
# df = pd.read_csv('main_pv/Databases/SolarStut.txt', delimiter=';')

# # Rename columns to be more descriptive
# # df.rename(columns={
# #     'MESS_DATUM': 'Datetime',
# #     'GS_10': 'Global_Irradiance',
# #     'DS_10': 'Diffuse_Irradiance',
# #     'SD_10': 'Sunshine_Duration',
# #     'LS_10': 'Longwave_Irradiance'
# # }, inplace=True)

# df.rename(columns={
#     'MESS_DATUM': 'Datetime',
# }, inplace=True)

# # Parse the Datetime column
# df['Datetime'] = pd.to_datetime(df['Datetime'], format="%Y%m%d%H%M")

# # Extract the hour from the `Datetime` column
# df['Hour'] = df['Datetime'].dt.hour

# # Filter for daytime hours (6 to 18 inclusive)
# daytime_df = df[(df['Hour'] >= 6) & (df['Hour'] <= 18)]

# # Define conversion factor from J/cm² per 10 min to W/m²
# conversion_factor = 10000/60/10

# # Convert relevant columns (assuming GS_10 and DS_10 are the solar irradiance columns)
# df['GS_10'] *= conversion_factor
# df['DS_10'] *= conversion_factor

# # Optionally rename columns for clarity
# df.rename(columns={'GS_10': 'GHI_wm2', 'DS_10': 'DHI_wm2'}, inplace=True)

# # Set reasonable thresholds for GHI and DHI (adjust as needed)
# ghi_threshold_low = 0
# ghi_threshold_high = 1200  # Assuming peak GHI rarely exceeds this for Germany
# dhi_threshold_low = 0
# dhi_threshold_high = 600   # Assuming peak DHI rarely exceeds this for Germany

# # Identify outliers based on thresholds
# ghi_outliers = (df['GHI_wm2'] < ghi_threshold_low) | (df['GHI_wm2'] > ghi_threshold_high)
# dhi_outliers = (df['DHI_wm2'] < dhi_threshold_low) | (df['DHI_wm2'] > dhi_threshold_high)

# # Plot histograms for GHI and DHI
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.hist(df['GHI_wm2'], bins=50, range=(ghi_threshold_low, ghi_threshold_high), color='skyblue', label='Valid Data')
# plt.hist(df.loc[ghi_outliers, 'GHI_wm2'], bins=50, range=(ghi_threshold_low, ghi_threshold_high), color='salmon', alpha=0.7, label='Outliers')
# plt.title('GHI Distribution with Outliers')
# plt.xlabel('GHI (W/m²)')
# plt.ylabel('Frequency')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.hist(df['DHI_wm2'], bins=50, range=(dhi_threshold_low, dhi_threshold_high), color='skyblue', label='Valid Data')
# # plt.hist(df.loc[dhi_outliers, 'DHI_wm2'], bins=50, range=(dhi_threshold_low, dhi_threshold_high), color='salmon',
# #          alpha=0.7, label='Outliers')
# plt.hist(df.loc[dhi_outliers, 'DHI_wm2'], bins=50, color='salmon',
#          alpha=0.7, label='Outliers')
# plt.title('DHI Distribution with Outliers')
# plt.xlabel('DHI (W/m²)')
# plt.ylabel('Frequency')
# plt.legend()

# plt.tight_layout()
# plt.show()

# list1 = ["nmksd", "pd,ak", "dskan"]
# list2 = ["hello", "woman", "cutie"]

# lister1 = [list1, list2]

# list3 = ["man", "pig", "bear"]
# list4 = ["but", "tien", "chiaotzu"]

# lister2 = [list3, list4]

# lister21 = [lister1, lister2]

# listerino = [listerlist for listerlist in lister21]

# print(listerino)
# listerino = [listerlist for listerlisto in listerino for listerlist in listerlisto]

# print(listerino)

# # Loaded variable 'df' from URI: h:\Studienarbeit\PycharmProjects\pythonProject\main_pv\Databases\Seasons\PCTs.csv
# import pandas as pd
# import numpy as np
# import datetime
# df = pd.read_csv(r'h:\\Studienarbeit\\PycharmProjects\\pythonProject\\main_pv\\Databases\\Seasons\\PCTs.csv', 
#                  delimiter = ";", index_col=[0,1,2], converters={'PctOnModule': lambda x: np.fromstring(x[1:-1], sep=',')})

# print(df)
# pctg_values_summer = df.at[("Summer", 180, 35), "PctOnModule"]
# pctg_values_winter = df.at[("Winter", 0, 60), "PctOnModule"]
# # print(f"Values for the summer are:\n {pctg_values_summer}\n and for the winter: \n {pctg_values_winter}")

# dict_summer = {hour: pct for hour, pct in enumerate(pctg_values_summer)}
# dict_winter = {hour: pct for hour, pct in enumerate(pctg_values_winter)}

# print(f"Values for the summer are:\n {dict_summer}\n and for the winter: \n {dict_winter}")

# print(f"First summer dawg is:\n {dict_summer[17]}\n and for the secon summer: \n {pctg_values_summer[17]}")

# date_start = np.datetime64(f"{2024}-01-01T00:00")
# date_end = np.datetime64(f"{2024}-01-01T12:00")

# date_start = np.datetime64(datetime.datetime(2024, 1, 1, 0))
# date_end = np.datetime64(datetime.datetime(2024, 1, 1, 14, 15))

# dif = date_end - date_start
# dif_in_h = dif/np.timedelta64(1, "h")

# print(13*dif_in_h)

# indies = pd.date_range(start=date_start, end=date_end, freq="10T").to_series()

# print(indies)
# def get_specific_weekly_stupid(e_car:ElCar):
#         """
#         :param e_car: The electric car.
#         return: Dataframe of The added charge demand caused by this car for the case of no smart meter and no PV. 
#         We're both stupid and cheap here. PV is for hippies and being smart is for nerds. This is for one car only. 
#         """
#         rest_energy = 0
#         values_car_temp = np.empty(shape=len(e_car.el_schedule))
#         indexes_car_temp = np.empty(shape=len(e_car.el_schedule), dtype=np.dtypes.DateTime64DType)
#         delt_10_min = np.timedelta64(10, "m")

#         for indiana, cycle in enumerate(e_car.el_schedule):
#             next_cycle = get_next_element(indiana, e_car.el_schedule)
#             max_charge_time = get_charge_time(cycle, next_cycle)
#             weekday_back, time_back = cycle.get_time_back()

#             energy_need = max(cycle.distance/e_car.efficiency + rest_energy, e_car.battery_cap)  # if the car goes below zero on fuel, 
#             # we assume they just charged somwehere else and came back empty. Battery level can't be negative.

#             demand = e_car.max_charge_pow  # we have no smart meters or other way to control the charging speed. 
#             # We're just stupidly sending electrons to their death. So it's just the max possible for the car.
#             # In reality, maybe the charging station doesn't support the max charging speed of the car. But that's an issue for another day.
#             # I've wasted long enough on this project.

#             charge_time = min(max_charge_time, energy_need/demand)  # Either we charge to full, 
#             # or we charge until we can't charge anymore because the owner is going on another trip before the battery is full.
            
#             np_charge_time_rounded = np.timedelta64(int(round(charge_time*6, 0)*10), "m")  # We round it because the database is in intervals of 10 minutes.

#             rest_energy = energy_need - demand * charge_time  # if not enough time to charge fully, the battery is left partly empty
#             start_datetime = np.datetime64(f"2024-01-0{weekday_back+1}T{time_back}")  # starting to charge directly after getting back from the current trip.
#             end_datetime = start_datetime + np_charge_time_rounded  # we assume time_back is a valid time and already rounded to the nearest 10 minutes.

#             new_inds = np.arange(start=start_datetime, stop=end_datetime, step=delt_10_min)  # these should be all the timestapmps where charging occurs.
#             indexes_car_temp[0] = new_inds  # adding our new timestamps into the indexes array.
#             index_len = len(new_inds)  # could just be one line with the next, but it would be a bit cluttered.
#             values_car_temp[0] = np.ones(shape=index_len, dtype=float)*demand  # the demand is constant and therefore can be applied to every timestamp in the indexes.
        
#         """
#         Pushing the dates that wrap around the week back to their rightful place.
#         """
#         threshold_date = pd.to_datetime('2024-01-08')

#         start_index = np.searchsorted(indexes_car_temp, threshold_date)  # I*ll be honest with you, I don't remember how this works.
#         if start_index < len(indexes_car_temp)-1:  # no clue.
#             indexes_car_temp[start_index:] -= np.datetime64(7, "D")  # It tries to push back anything past- 
#             # a week to the start of the month, but does it work? no idea.

#         """
#         creating the dataframe.
#         """
#         indexes_car_temp = [[d.day-1, d.time()] for d in indexes_car_temp]  # the dataframe has the indexes day of week and time. 
#         # 2024 starts on monday, so we push back the day of month by one and get the day of the week.

#         index = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)
        
#         db_week_car_update = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=values_car_temp)  # we use update because- 
#         # otherwise the database should be full of zeros when the car isn't charging.
#         return db_week_car_update

