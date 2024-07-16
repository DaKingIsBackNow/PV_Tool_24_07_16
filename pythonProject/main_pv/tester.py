import numpy as np
import pandas as pd
from main_pv.classes import *
import time


# np_charge_time = np.timedelta64(rounded_charge_time, "m")
# rest_time = charge_time%(1/6)
# last_spread_demand = demand*rest_time*6  # kW
# print(rest_time, last_spread_demand, demand) 
# rounded_charge_time = int(round(charge_time*6, 0)*10)  # minutes

# charge_time = 2.3498321  # hours
# max_battery_cap = 50  # kWh
# battery_level_cur = 40  # kWh
# max_charge_power = 8  # kW
# energy_need = max_battery_cap - battery_level_cur  # kWh
# demand = min(max_charge_power, energy_need/charge_time)  # kW

# print(demand)

# datey_time = np.datetime64(f"{2024}-01-01T00:00")
# deltie = np.timedelta64(10, "m")

# date_start = datey_time
# date_stop = date_start + np.timedelta64(7, "D")
# indies = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))
# data = np.empty(dtype=float, shape=len(indies))

# indies = [[d.day-1, d.time()] for d in indies]
# values = np.zeros(shape=len(indies))

# print(values)

# heaty = HeatingSystem(0, 0, 2, 1998, 32, 10000)


# db_year = calc._get_yearly_smart_no_pv(2021)
# print(db_year.head(10))

time_start = time.time()
Schedulino = np.empty(shape=5, dtype=ElCarCycle)
for i in range(5):
    departure = datetime.time(hour=i*5, minute=i*10)
    weekday = i
    time_away = datetime.timedelta(hours=3+i)
    distance = 20+5*i
    Schedulino[i] = ElCarCycle(departure, time_away, distance, weekday)

Schedulino = Schedulino.tolist()
# for sched in Schedulino:
#     print(sched)
Caroline = ElCar(50, 8, 5, Schedulino)
heaty = HeatingSystem(0, 0, 2, 1998, 32, 10000)
pv_sister = PVSystem(8, 2005, 180, 35)
homey = Building(400, 200, 5, 600, 0.95, 3, 1967, 4, 60, [heaty], [Caroline], True, 8)
calc = EnergyManagement([homey])

datetime_start = np.datetime64("2024-01-01T00:00")
datetime_end = np.datetime64("2024-01-01T17:00")
max_solar = calc._get_simple_smart_maxsolar(datetime_start, datetime_end, True)
print(f"{max_solar} kWh potential in that time.")

time_end = time.time()
print(f"time needed to run the bitch is {time_end-time_start} seconds")

