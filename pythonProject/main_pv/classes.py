import pandas as pd
import datetime
import logging
import numpy as np
import os 
from main_pv.standard_variables import *
import main_pv.solar_functions as solar_functions
from numba import njit

DEBUGGING = False

if DEBUGGING:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

@njit
def get_first_index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    # If no item was found return None, other return types might be a problem due to
    # numbas type inference.
    return None
class HeatingSystem:
    Main_SysTypes_Names = pd.read_csv(DIR_PATH + MAINTYPES_PATH, delimiter=DELIMITER, decimal=",", index_col=0)
    SubSysTypes = pd.read_csv(DIR_PATH + SUBTYPES_PATH, delimiter=DELIMITER, index_col=0)
    EngSource = pd.read_csv(DIR_PATH + ENGSRC_PATH, delimiter=DELIMITER, index_col=0)
    Subsidies = pd.read_csv(DIR_PATH + SUBSID_PATH, delimiter=DELIMITER, index_col=0)
    SubMainPairings = pd.read_csv(DIR_PATH + SUB_MAIN_PAIRINGS_PATH, delimiter=DELIMITER, index_col=0)

    def __init__(self, sys_type_ind: int, sub_sys_type_ind: int, eng_source: int, year: int, power: float,
                 cost: float = 0., share: float = 0):
        """
        :param sys_type_ind: index for the system type
        :param sub_sys_type_ind: index for the sub-system type
        :param eng_source: index for the energy source
        :param year: construction year
        :param power: p in kW
        :param share: how much of the total heat demand of the circuit is this system covering. from 0 to 1
        """
        if sys_type_ind not in self.Main_SysTypes_Names.index:
            raise ValueError("There is no system type with this index")
        if sub_sys_type_ind not in (self.SubMainPairings.loc[self.SubMainPairings[MAIN_INDEX_STRING] ==
                                                         sys_type_ind]).index:
            raise ValueError("This Subsystem Index is invalid for this main system")
        if eng_source not in (self.SubSysTypes.loc[self.SubSysTypes[SUBTYPE_INDEX_STRING] ==
                                                   sub_sys_type_ind])[ENG_INDEX_STRING].values:
            raise ValueError("This energy index is invalid for this sub system")
        if not(0 <= share <= 1):
            raise ValueError("Share must be between 0 and 1.")
        
        self.sys_type_ind = sys_type_ind
        self.sub_sys_type_ind = sub_sys_type_ind
        self.eng_source_ind = eng_source
        self.con_year = year
        self.power = power
        self.cost = cost
        row = self.SubSysTypes.loc[self.SubSysTypes[SUBTYPE_INDEX_STRING] == sub_sys_type_ind]
        row = row.loc[row[ENG_INDEX_STRING] == eng_source]
        self.help_electricity = row[HELP_ENG_STRING].values[0] * self.power
        self.share = share

    def __str__(self):
        return (f"Type: {(self.SubSysTypes.iloc[[self.sub_sys_type_ind]])['Name'].values[0]}, "
                f"Source: {(self.EngSource.iloc[[self.eng_source_ind]])['Name'].values[0]}"
                f", Year: {self.con_year}, Power: {self.power} kW")
        # return f"Type: {(self.SysTypes.iloc[[self.sys_type_ind]])['Name'].values[0]}"

    def get_base_subsidy(self):
        sub_id = self.SubSysTypes.loc[(self.SubSysTypes[SUBTYPE_INDEX_STRING] == self.sub_sys_type_ind) & (
                self.SubSysTypes[ENG_INDEX_STRING] == self.eng_source_ind
        )][SUBSINDEX_STRING].values[0]
        return self.Subsidies.iloc[[sub_id]]["BasePer"].values[0]

    def get_subsidy_description(self):
        sub_id = self.SubSysTypes.loc[(self.SubSysTypes[SUBTYPE_INDEX_STRING] == self.sub_sys_type_ind) & (
                self.SubSysTypes[ENG_INDEX_STRING] == self.eng_source_ind
        )][SUBSINDEX_STRING].values[0]
        return self.Subsidies.iloc[[sub_id]]["Description"].values[0]

    def get_eng_source(self):
        return self.EngSource.iloc[[self.eng_source_ind]]["Name"].values[0]

    def get_efficiency(self, flow_temp: int = 70, ret_temp: int = 55):
        df_sub = self.SubSysTypes
        row = df_sub.loc[(df_sub[SUBTYPE_INDEX_STRING] == self.sub_sys_type_ind) & (df_sub[ENG_INDEX_STRING] == self.eng_source_ind)]
        row = row.reset_index()
        eff = row.at[0, EFF_STRING]

        if self.Main_SysTypes_Names.at[self.sys_type_ind, "Name"] == HEAT_PUMP_STRING:  # for heat pumps we want to account for lower flow temps than 70 and raise the efficiency.
            max_factor = MAX_HEAT_PUMP_FLOW_TEMP_FACTOR
            factor = 1 + (max_factor-1)*(flow_temp - DEF_FLOW_TEMP)/(DEF_MIN_FLOW_TEMP-DEF_FLOW_TEMP)
            eff = eff*factor

        return eff

class HeatingCircuit:

    def __init__(self, flow_temp: int, ret_temp: int, share: float, heating_systems: list[HeatingSystem]):
        """

        :param flow_temp: the flow temperature of the heating circuit
        :param ret_temp: the return temperature of the heating circuit
        :param share: the share of the whole heat demand this circuit supplies (between 0 und 1)
        :param heating_systems: heating_systems
        """
        if not(0 <= share <= 1):
            raise ValueError("Share must be between 0 and 1.")
        self.flow_temp = flow_temp
        self.ret_temp = ret_temp
        self.share = share
        self.heating_systems = heating_systems


class ElCarCycle:
    def __init__(self, departure: datetime.time, time_away: datetime.timedelta, distance: float, weekday: int):
        """

        :param departure: departure time i.e. 20:30:00
        :param time_away: how long until the car is back in the building
        :param distance: in km
        :param weekday: 0 = Monday 6 = Sunday
        """
        self.departure_time = departure
        self.time_away = time_away
        self.distance = distance
        self.weekday_depart = weekday

    def __str__(self) -> str:
        return f"Elcarcycle leaving at {self.departure_time} day numnber: {self.weekday_depart} distance: {self.distance} km, time away: {self.time_away}"

    def get_time_back(self):
        """
            :returns: Tuple (the weekday, time back)
        """
        vir_time_depart = datetime.datetime(year=2024, month=1, day=self.weekday_depart + 1, hour=self.departure_time.hour,
                                            minute=self.departure_time.minute)  # let's pretend we're in the
        # beginning of 2024 to work with datetime objects. The first of January was even a monday,
        # which works great with our weekday convention.

        vir_time_back = vir_time_depart + self.time_away

        return vir_time_back.weekday(), vir_time_back.time()


def get_solar_db() -> pd.DataFrame:
    path = DIR_PATH + SOLPATH

    return pd.read_csv(path, delimiter=DELIMITER, index_col=0)

def get_el_demand_dataframe() -> pd.DataFrame:
    path = DIR_PATH + ELDEMAND_PATH
    return pd.read_csv(path, delimiter=DELIMITER, index_col=2)

def get_pct_db():
    path = DIR_PATH + PCT_PATH
    return pd.read_csv(path, 
                        delimiter = ";", 
                        index_col=[0,1,2], 
                        converters={PERCENT_STRING: lambda x: np.fromstring(x[1:-1], sep=',')})

def get_charge_time(current_cycle: ElCarCycle, next_cycle: ElCarCycle):
    """
    Args:
        :param current_cycle: Current drive with the E-Car.
        :param next_cycle: Next drive with the E-Car

    Returns: the time difference between when the car is back home after the first drive and the
        departure time of the next in hours.
    """
    weeday_now = current_cycle.weekday_depart
    weeday_next = next_cycle.weekday_depart
    time_departure = (current_cycle.departure_time.hour + current_cycle.departure_time.minute / 60) / 24
    time_delta = current_cycle.time_away.days + current_cycle.time_away.seconds / 3600 / 24  # days returns the rounded
    # down number of days away and the seconds returns the rest of the seconds after that amount of days. That's the
    # reason for this stupid calculation.
    decimal_time_back = time_departure + time_delta

    # check if we crossed midnight
    if decimal_time_back >= 1:
        decimal_time_back -= 1
        weeday_now += 1
    time_next = next_cycle.departure_time
    decimal_time_next = (time_next.hour + time_next.minute / 60) / 24

    time_dif = weeday_next - weeday_now + decimal_time_next - decimal_time_back

    # check if we crossed a week
    if time_dif < 0:
        time_dif += 7

    return time_dif * 24

def get_next_element(ind: int, lister: list):
    """
    Args:
        :param ind: Current index.
        :param lister: list of elements.

    Returns: the next element of the list.
    """
    if ind == len(lister) - 1:
        return lister[0]
    return lister[ind + 1]


def get_weekly_indies(time_delta: datetime.timedelta):
    """
    :param time_delta: The time interval of the weekly schedule. For example 10 minutes. The resolution.
    :return: the indexes for a weekly schedule with the time intervals of time_delta in the form of a list of Tuples
    [(weekday, time of day), ...]
    """
    datey_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    deltie = time_delta
    indies = []
    while datey_time.day < 8:
        indies.append((datey_time.weekday(), datey_time.time()))
        datey_time += deltie
    return indies


def get_times_next_cycle_weekly(next_cycle: ElCarCycle):
    """
    This function is only for weekly schedules and uses 2024 as a reference year. We add one to the weekday because
    it starts at 0 and 2024 started on a monday.
    :param next_cycle: The next driving trip of the car.
    :return: a tuple (datetime of next return from trip, datetime of next departure)
    """
    day_back, time_back = next_cycle.get_time_back()
    next_time_back = np.datetime64(datetime.datetime(year=2024, month=1, day=day_back+1,
                                       hour=time_back.hour, minute=time_back.minute))
    next_time_departure = np.datetime64(datetime.datetime(year=2024, month=1, day=next_cycle.weekday_depart + 1,
                                            hour=next_cycle.departure_time.hour, minute=next_cycle.departure_time.minute))
    return next_time_back, next_time_departure


# def get_winter_avg():
#     path = DIR_PATH + "/Databases/Seasons/Winter.csv"
#     df = pd.read_csv(path)
#     return df

# def get_summer_avg():
#     path = DIR_PATH + "/Databases/Seasons/Summer.csv"
#     df = pd.read_csv(path)
#     return df


class PVSystem:
    def __init__(self, max_power: float|int=0, con_year: int=2000, azimuth: float|int=180, angle: float|int=0) -> None:
        """
        :param max_power: the maximum power in kW (ie. 8).
        :param con_year: construction year (ie. 2004).
        :param azimuth: module azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
        :param angle: the angle from the horizontal in degrees (ie. 35).

        """
        self.max_power = max_power
        self.con_year = con_year
        self.azimuth = azimuth
        self.angle = angle
    
    def get_pv_generation_year(self) -> np.ndarray:
        df_sol = get_solar_db()
        pv_pow = self.max_power

        pctg = solar_functions.get_pctgs(self.azimuth, self.angle, df_sol)

        return pctg*pv_pow
        


class ElCar:
    def __init__(self, battery_cap: float, max_charge_pow: float, efficiency: float, el_schedule: list[ElCarCycle]):
        """

        :param battery_cap: capacity in kWh
        :param max_charge_pow: max charging power in kW
        :param efficiency: in km/kWh, usually around 5.
        :param el_schedule: list of E-Car cycles
        """
        self.battery_cap = battery_cap
        self.max_charge_pow = max_charge_pow
        self.efficiency = efficiency
        self.el_schedule = el_schedule
        self.el_schedule = sorted(self.el_schedule, key=lambda cycle: (cycle.weekday_depart, cycle.departure_time))

        """
        Calculation internal variables.
        """
        self.battery_level = self.battery_cap
        self.demand = 0
        self.car_home = True
        self.cur_cycle = None
        self.next_cycle = None
        self.next_time_back = self.next_time_departure = None
        self.cyc_ind = 0
        self.charge_plan = None
        self.df_schedule = None
        self.initiate_eng_need_weekly_schedule()

    def initiate_eng_need_weekly_schedule(self):
        if self.df_schedule is not None:
            return
        deltie = TIME_DIF
        date_start = np.datetime64(f"{2024}-01-01T00:00")  # 2024 cause it started on Monday
        date_end = np.datetime64(f"{2024}-01-08T00:00")  # 2024 cause it started on Monday
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_end, step=deltie, dtype=np.datetime64))
        values = np.zeros(len(times))
        indexes_car_temp = np.empty(shape=0, dtype=np.dtypes.DateTime64DType)
        values_car_temp = np.empty(shape=0, dtype=bool)

        e_car = self
        for indiana, cycle in enumerate(e_car.el_schedule):
            next_cycle = get_next_element(indiana, e_car.el_schedule)
            charge_time = get_charge_time(cycle,next_cycle)
            weekday_back, time_back = cycle.get_time_back()
            energy_need = min(cycle.distance/e_car.efficiency, e_car.battery_cap)

            np_charge_time_rounded = np.timedelta64(int(round(charge_time*6, 0)*10), "m")  # We round it because the database is in intervals of 10 minutes.

            start_datetime = np.datetime64(f"2024-01-0{weekday_back+1}T{time_back}")  # starting to charge directly after getting back from the current trip.
            end_datetime = start_datetime + np_charge_time_rounded  # we assume time_back is a valid time and already rounded to the nearest 10 minutes.

            new_inds = np.arange(start=start_datetime, stop=end_datetime, step=deltie)  # these should be all the timestapmps where charging occurs.
            indexes_car_temp = np.append(indexes_car_temp, new_inds)  # adding our new timestamps into the indexes array.
            index_len = len(new_inds)  # could just be one line with the next, but it would be a bit cluttered.
            values_car_temp =  np.append(values_car_temp, np.tile(energy_need, index_len))  # the demand is constant and therefore can be applied to every timestamp in the indexes.

        
        """
        Pushing the dates that wrap around the week back to their rightful place.
        """
        threshold_date = pd.to_datetime('2024-01-08')
        indexes_car_temp = np.array(indexes_car_temp, dtype=np.dtypes.DateTime64DType)
        start_index = np.searchsorted(indexes_car_temp, threshold_date)  # I*ll be honest with you, I don't remember how this works.
        if start_index < len(indexes_car_temp)-1:  # no clue.
            week_offset = np.timedelta64(7, "D").astype('datetime64[D]')
            indexes_car_temp[start_index:] = indexes_car_temp[start_index:] - week_offset  # It tries to push back anything past- 
            # a week to the start of the month, but does it work? no idea.

        """
        creating the dataframe.
        """
        index = pd.DatetimeIndex(times, name=TIMESTAMP_STRING)
        update_index = pd.DatetimeIndex(indexes_car_temp, name=TIMESTAMP_STRING)

        self.df_schedule = pd.DataFrame(columns=[EL_CAR_ENERGY_NEEDED_STRING], index=index, data=values)
        df_update = pd.DataFrame(columns=[EL_CAR_ENERGY_NEEDED_STRING], index=update_index, data=values_car_temp)
        self.df_schedule.update(df_update)

    def reset_vars(self):
        """
        Resets the schedule since the schedule calculation does nothing if one already exists.
        """
        self.el_schedule=None

    def is_in_garage(self, timestamp: np.datetime64) -> bool:
        """
        return: True if the car is in the garage and False if the car is literally anywhere else.
        """
        self.initiate_eng_need_weekly_schedule()
        date_time = pd.to_datetime(timestamp)
        week_day, timey = date_time.weekday(), date_time.time()
        fake_datetime = np.datetime64(f"{2024}-01-0{week_day+1}T{timey}")  # 2024 cause it started on Monday

        is_there = self.df_schedule.at[fake_datetime, EL_CAR_ENERGY_NEEDED_STRING] != 0

        return is_there
    
    def _get_energy_needed(self, timestamp: np.datetime64) -> float:
        self.initiate_eng_need_weekly_schedule()
        date_time = pd.to_datetime(timestamp)
        week_day, timey = date_time.weekday(), date_time.time()
        fake_datetime = np.datetime64(f"{2024}-01-0{week_day+1}T{timey}")  # 2024 cause it started on Monday

        return self.df_schedule.at[fake_datetime, EL_CAR_ENERGY_NEEDED_STRING]

    def _get_charge_time_left(self, timestamp: np.timedelta64) -> float:
        """
        Returns the charge time left until next departure in hours.
        """
        self.initiate_eng_need_weekly_schedule()
        date_time = pd.to_datetime(timestamp)
        week_day, timey = date_time.weekday(), date_time.time()
        fake_datetime = np.datetime64(f"{2024}-01-0{week_day+1}T{timey}")  # 2024 cause it started on Monday
        df = self.df_schedule

        mask = (df.index > fake_datetime) & (df[EL_CAR_ENERGY_NEEDED_STRING] == 0)
        next_departure_timestamp = df.loc[mask].index[0]  # Get first match
        time_diff_seconds = (next_departure_timestamp - fake_datetime).total_seconds()

        return time_diff_seconds / 3600

    # def _get_advanced_smart_demand(self, pv_power: float, db: pd.DataFrame):
    #     """
    #         A function for the summer demand in case of smart meters in the advanced method that takes the solar data
    #         into account.
    #     :param pv_power: max PV power in kW.
    #     :param db: a database with the solar radiation data throughout the year.
    #         Returns: The summer charging schedule of the electric car, prioritizing times with solar radiation.
    #     """
    #
    #     return "Not happening dawg."
    #
    #     Date = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
    #     delta = datetime.timedelta(minutes=10)
    #     rest_energie = 0
    #     for ind, cycle in enumerate(self.el_schedule):
    #         energy_need = cycle.distance/self.efficiency + rest_energie  # in kWh
    #         next_cycle = get_next_element(ind, self.el_schedule)
    #         time_charge = get_charge_time(cycle, next_cycle)  # in hours
    #         power_full_batt = energy_need/time_charge  # in kW
    #
    #         if power_full_batt > self.max_charge_pow:
    #             cur_power = self.max_charge_pow
    #             rest_energie = energy_need - self.max_charge_pow*time_charge
    #         else:
    #             cur_power = power_full_batt
    #         smart_power = min(power_full_batt, self.max_charge_pow)
        
    def get_db_week_thingy_summer(self):
        """
            Creates a weekly schedule of electric charging and returns the dataframe. Using 2024 because it started
            on a monday.The year doesn't get saved later and plays no part in the returned database.
        """

        """
        Initializing shit.
        """
        solar = 0
        self.el_schedule = sorted(self.el_schedule, key=lambda cyc: (cyc.weekday_depart, cyc.departure_time))
        self.cur_cycle = self.el_schedule[-1]
        self.next_cycle = self.el_schedule[0]
        self.cyc_ind = 0
        self.battery_level = self.battery_cap

        self.next_time_back, self.next_time_departure = get_times_next_cycle_weekly(self.next_cycle)

        datey_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
        deltie = datetime.timedelta(minutes=10)
        data = []
        indies = []
        self.demand = 0
        self.car_home = True

        """
        Going through the week in ten minute steps, starting from midnight going into Monday.
        """

        while datey_time.day < 8:
            indies.append((datey_time.weekday(), datey_time.time()))

            if datey_time == self.next_time_departure:
                self._smart_smart_departure()
            if datey_time == self.next_time_back:
                self._smart_smart_got_back()
            if self.car_home and self.battery_level != self.battery_cap:
                self._smart_smart_charge()

            data.append(self.demand)
            datey_time += deltie

        index = pd.MultiIndex.from_tuples(indies, names=WEEK_INDNAMES)
        db_week = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=data)
        return db_week

    def _get_yearly_stupid_demand(self, year):
        """
        :return: the full car charging schedule for the case of no smart meter.
        """
        db_week = self.get_db_weekly_stupid_demand()
        datey_time = datetime.datetime(year=year, month=1, day=1, hour=0, minute=0)
        deltie = datetime.timedelta(minutes=10)
        data = []
        indies = []

        while datey_time.year == year:
            indies.append((datey_time.date(), datey_time.time()))
            data.append(db_week.loc[(datey_time.weekday(), datey_time.time())].values[0])
            datey_time += deltie

        index = pd.MultiIndex.from_tuples(indies, names=YEAR_INDNAMES)
        db_year = pd.DataFrame(index=index, columns=[ELDEMAND_STRING], data=data)
        return db_year

    def _stupid_departure(self) -> None:
        """
        Stupid car goes on a trip. So we update the status of it being home. The next driving cycle is becoming the
        current. And the next is now the one after that.
        """
        self.demand = 0
        self.car_home = False
        self.cur_cycle = self.next_cycle
        self.next_cycle = get_next_element(self.cyc_ind, self.el_schedule)
        self.cyc_ind += 1

    def _stupid_got_back(self) -> None:
        """
        Stupid car gets back from trip. So we update the battery level by the energy the trip needed.
        """
        self.battery_level = max(self.battery_level - self.cur_cycle.distance / self.efficiency, 0)
        self.demand = self._get_current_stupid_demand(self.battery_level)
        self.next_time_back, self.next_time_departure = get_times_next_cycle_weekly(self.next_cycle)
        self.car_home = True

    def _stupid_charge(self) -> None:
        """
        Stupid car is home and the battery is not full yet, so we charge and apply some logic on the bitch.
        """
        self.battery_level += self.demand / 6  # demand is in kW and battery in kWh.
        # Since the time steps are ten minutes, we divide by 6.
        if round(self.battery_level, 2) == self.battery_cap:  # if the battery is now full, the demand is 0.
            # We round it in case of rounding errors.
            self.battery_level = self.battery_cap
            self.demand = 0
        else:  # if the battery is not full, we find the next demand for the next ten minutes.
            self.demand = self._get_current_stupid_demand(self.battery_level)

    def get_db_weekly_stupid_demand(self):
        """
        Creates a weekly schedule of electric charging and returns the dataframe. Using 2024 because it started on a monday.
        The year doesn't get saved later and plays no part in the returned database.
        """

        """
        Initializing shit.
        """
        self.el_schedule = sorted(self.el_schedule, key=lambda cyc: (cyc.weekday_depart, cyc.departure_time))
        self.cur_cycle = self.el_schedule[-1]
        self.next_cycle = self.el_schedule[0]
        self.cyc_ind = 0
        self.battery_level = self.battery_cap

        self.next_time_back, self.next_time_departure = get_times_next_cycle_weekly(self.next_cycle)

        datey_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
        deltie = datetime.timedelta(minutes=10)
        data = []
        indies = []
        self.demand = 0
        self.car_home = True

        """
        Going through the week in ten minute steps, starting from midnight going into Monday.
        """

        while datey_time.day < 8:
            indies.append((datey_time.weekday(), datey_time.time()))

            if datey_time == self.next_time_departure:
                self._stupid_departure()
            if datey_time == self.next_time_back:
                self._stupid_got_back()
            if self.car_home and self.battery_level != self.battery_cap:
                self._stupid_charge()

            data.append(self.demand)
            datey_time += deltie

        index = pd.MultiIndex.from_tuples(indies, names=WEEK_INDNAMES)
        db_week = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=data)
        return db_week

    def _get_current_stupid_demand(self):
        """
        :return: The current demand for a non-smart meter case.
        """
        energy_need = self.battery_cap - self.battery_level
        demand = min(self.max_charge_pow, energy_need * 6)  # times six to get the kW needed to cover the
        # energy needed in ten minutes. In reality the car would charge with max power for two minutes for
        # example, but since we work with ten minute values, we spread the energy demanded over the full ten
        # minutes.
        return demand

    def get_yearly_demand(self, smart: bool, pv_max: float = 0, advanced: bool = False, year: int = 2024):
        """
        :param smart: a boolean paremeter for the case of a smart meter. True if smart meter installed. False if not.
        :param advanced: a boolean paremeter for the advanced method in case of a smart meter.
                         True if the advance method is desired. False if not.
        :param pv_max: the maximum pv power of the building.
        :param year: Let's be honest. No one will ever use this parameter.
        """
        if not smart:
            return self._get_yearly_stupid_demand(year)
        if pv_max == 0:
            return
        if advanced:
            pass  # return advanced winter plus advanced
        return self._get_yearly_simple_smart_demand(year)

    def add_cycle(self, departure, time_away, distance, weekday):
        """
            Adds a drive cycle to the car schedule.
        :param departure: The time of leaving the house.
        :param time_away: The length of time the car will be away.
        :param distance: The complete distance in km in the trip.
        :param weekday: The weekday 0 = Monday 6 = Sunday
        """
        self.el_schedule.append(ElCarCycle(departure, time_away, distance, weekday))
        self.el_schedule = sorted(self.el_schedule, key=lambda cycle: (cycle.weekday_depart, cycle.departure_time))


class HeatPump(HeatingSystem):

    def __init__(self, sys_type_ind: int, sub_sys_type_ind: int, eng_source: int, year: int, power: float,
                 biv_type: int, biv_point_1: float, biv_point_2: float, cops: pd.DataFrame, help_electricity=0.):
        """

        :param sys_type_ind:
        :param sub_sys_type_ind:
        :param eng_source:
        :param year:
        :param power:
        :param biv_type: 0 for alternate 1 for parallel and 2 for part parallel
        :param biv_point_1: for alternate this is the lowest temperature where the heat pump still works,
        for parallel and part parallel this is where the other heating device starts working.
        :param biv_point_2: only exists for part parallel, the point where the heat pump stops working.
        """
        super().__init__(sys_type_ind, sub_sys_type_ind, eng_source, year, power, help_electricity)
        self.biv_type = biv_type
        self.biv_point_1 = biv_point_1
        self.biv_point_2 = biv_point_2


class RoofObstruction:
    def __init__(self, area: float, type:str = "Undefined") -> None:
        """
        could be a window or a chimney, or garden or whatever makes it unsuitable for pv.
        """
        self.area = area


class Roof:
    def __init__(self, real_area: float, azimuth: float, angle: float, roof_objects: list[RoofObstruction], is_main_roof: bool = True, total_area: float = None) -> None:
        """
        A roof area.
        real_area: the actual area of the roof. Including windows and chimneys but not dormers.
        azimuth: azimuth in degrees North = 0, West = - 90, East = + 90, South = 180.
        angle: The angle of the roof over the horizon. 0 is flat, 90 is a wall.
        roof_objects: A list of windows and chimneys and all that.
        is_main_roof: bool describing if the roof is a main roof or just a dormer.
        total_area: The theoretical area of the roof if no dormers exist. Should be the same or higher than real area. 
        For dormers this term doesn't exist.
        """
        self.real_area = real_area
        self.azimuth = azimuth
        self.angle = angle
        self.roof_obstructions = roof_objects
        self.is_main_roof = is_main_roof
        self.total_hypothetical_area = self.real_area if total_area is None else total_area
    
    def get_netto_area(self):
        """
        Returns the area that can be used for pv since no window or other object is there.
        """
        if self.roof_obstructions is None:
            return self.real_area
        return self.real_area - sum([roofie.area for roofie in self.roof_obstructions])

    def is_solar_suitable(self):
        """
        Returns if this roof area is suitable for solar use.
        """
        is_south = (self.azimuth <= -90 or self.azimuth >= 90)
        is_flat = self.angle < 20
        is_too_steep = self.angle > 60
        solar_suitable = (is_south or is_flat) and (not is_too_steep)
        return solar_suitable
    
    def get_pv_standard(self):
        """
        returns the squared meters of pv needed for the standard method. Sixty percent of the total area if the roof is suitable and none if it isn't.
        """
        if not self.is_solar_suitable() or not self.is_main_roof:
            return 0
        ret_val = PV_STANDARD_METHOD_FACTOR*self.total_hypothetical_area
        return ret_val
    
    def get_pv_advanced(self):
        """
        Returns the amount of pv needed in squared meters according to the advanced method. seventy five percent of the real netto area.
        """
        if not self.is_solar_suitable():
            return 0
        net_area = self.get_netto_area()
        ret_val = PV_ADVANCED_METHOD_FACTOR*net_area
        return ret_val

    def get_pv_simple(self):
        """
        Returns the pv needed based on the basic method. Doesn't matter if the roof is south, north has windows or any of that.
        Result is in kWp.
        """
        if not self.is_main_roof:
            return 0
        built_area = self.total_hypothetical_area*np.cos(np.deg2rad(self.angle))
        ret_val = built_area*PV_SIMPLE_METHOD_FACTOR
        return ret_val
    


class Building:
    

    def __init__(self, h_a: float, liv_area: float, residents: int, air_vol: float, ht: float, res_units: int,
                 con_year: int, large_el_devices: int, dw_temp: int, heating_circuits: list[HeatingCircuit], e_cars: list[ElCar],
                 smartmeter: bool, pv_sytem: list[PVSystem], base_el_demand: int = None, heat_demand: int = None, 
                 battery_cap: float = 0, roofs: list[Roof] = None, roof_change: bool = True, heat_change: bool = True, 
                 is_in_bw: bool = True, solar_thermal_area: float = 0, have_renovation_roadmap: bool = False) -> None:
        """

        :param h_a: thermal hull area m²
        :param v: thermal heated volume m³
        :param ht: average heat loss in W/m²K
        :param con_year: construction year
        :param liv_area: living area in m²
        :param res_units: number of residential units
        :param residents: number of residents
        :param large_el_devices: such as fridge, washing machine, dishwasher
        :param dw_temp: temperature of the hot drinking water
        :param heating_circuits: a list of the heating circuits
        :param e_cars: a list of the electric cars
        :param smartmeter: True if a smart meter is installed and false if not
        :param pv_sytem: The pv system.
        :param base_el_demand: The yearly base electricity consumption, excluding heating or electric cars. in kWh
        :param heat_demand: The yearly consumption for heating in kWh.
        :param battery_cap: The capacity of the battery in kWh.
        :param roofs: The roofs of the building.
        :param roof_change: True if the roof is being insulated or changed soon or has recently. False if roof remains as is.
        :param heat_change: True if the heating systems of the building are being changed soon and false if not.
        :param is_in_bw: True if the building is in Baden Württemberg, false if not.
        :param solar_thermal_area: The area of solar thermal panels the building has installed.
        :param have_renovation_roadmap: A bool saying if the building has a renovation roadmap or not.
        """
        self.large_el_devices = large_el_devices
        self.res_units = res_units
        self.residents = residents
        self.liv_area = liv_area
        self.hull_area = h_a
        self.air_volume = air_vol
        self.ht = ht
        self.con_year = con_year
        self.e_cars = e_cars
        self.smartmeter = smartmeter
        self.dw_temp = dw_temp
        self.heating_circuits = heating_circuits
        self.heat_rep_bonus = False
        self.pv_system = pv_sytem

        self.standard_room_temp = STANDARD_ROOM_TEMP
        self.outside_air_temp = STANDARD_OUTSIDE_TEMP

        self.air_exchange_rate = None
        self.total_volume = None
        self.max_heat_demand = None
        self.warm_water_demand = None
        self.help_energy = None
        self.battery_cap = battery_cap

        self.electricity_share_heating = None
        self.average_el_heat_eff = None

        self.base_el_consumption = None
        self.heat_consumption = None

        self.base_el_consumption = self.get_yearly_base_el_demand() if base_el_demand is None else base_el_demand
        self.heat_consumption = self.get_yearly_base_el_demand() if heat_demand is None else heat_demand

        self.roof_change = roof_change
        self.heat_change = heat_change
        self.is_in_bw = is_in_bw
        self.solar_thermal_area = solar_thermal_area
        self.have_renovation_roadmap = have_renovation_roadmap

        if roofs is None:
            """
            If no roofs are given, we make some assumptions. It's better to know the roofs.
            """
            roof_south = Roof(real_area=self.liv_area/2, azimuth=180, angle=35, roof_objects=None, is_main_roof=True)
            roof_north = Roof(real_area=self.liv_area/2, azimuth=0, angle=35, roof_objects=None, is_main_roof=True)
            roofs = [roof_north, roof_south]
        
        self.roofs = roofs


    def set_new_heating_system(self, new_heating: HeatingSystem):
        self.heating_circuits = [new_heating]
    
    def get_pv_needed(self, new_solar_thermal_planned: float = 0) -> float:
        """
        Returns the amount of pv needed if the roof will be changed.
        """
        """
        Case not in Baden Württemberg.
        """
        if not self.is_in_bw:
            return 0
        """
        Case roof unchanged.
        """
        if not self.roof_change:
            return self._get_pv_e_waerm_g(new_solar_thermal_planned)
            
        """
        Case roof changed. You technically need the max between the e-wärme-g and pv-pflicht verordnung. 
        But the roof change method pretty much always gives a higher mandatory size.
        """
        return self._get_pv_mandatory_roof_change(new_solar_thermal_planned)

    def _get_pv_e_waerm_g(self, new_solar_thermal_planned: float = 0) -> float:
        """
        Returns the amount of pv needed for the E-WärmeG in kWp. 
        Does not check if building is in BW or if building uses a renewable heating system like heat pump or pellet boiler.
        """
        solar_thermal_total = new_solar_thermal_planned + self.solar_thermal_area
        
        if not self.heat_change:  # if not changing the heating system, no requirement.
                return 0
        
        need_factor = E_WAERM_G_FACTOR_NEED  # fetching how much percent renewable we need.
        factor_have = 0  # starting to sum our percent renewable outside of pv.

        if self.have_renovation_roadmap:
            factor_have += E_WAERM_FACTOR_RENOVATION_ROADMAP  # if there is a renovation roadmap, we reduce the requirements.

        if self.res_units <= 2:  # the factor for solar thermal depends on the number of residential units.
            sol_therm_factor = E_WAERM_G_SOLTHERM_FACTOR_ONE_TWO_UNITS
        else:
            sol_therm_factor = E_WAERM_G_SOLTHERM_FACTOR_MORE_UNITS

        factor_have += solar_thermal_total/(sol_therm_factor*self.liv_area)  # adjusting for solar thermal.
        if factor_have >= need_factor:  # if we already have enough renewables, we don't need any pv.
            pv_need = 0
            return 0
        
        pv_need_factor = need_factor - factor_have  # what we need for pv is what we need in total minus what we have through other means.

        pv_need = pv_need_factor*(E_WAERM_G_PV_FACTOR*self.liv_area)  # parentheses really not necessary, but whatever.

        return pv_need

    def _get_pv_mandatory_roof_change(self, new_solar_thermal_planned):
        """
        Returns the amount of pv needed in kWp for the building if the roof is changed and the building resides in Baden Württemberg.
        """
        solar_thermal_total = new_solar_thermal_planned + self.solar_thermal_area

        solar_thermal_pv_equivalent = (solar_thermal_total)/SOLAR_THERMAL_FACTOR
        
        """
        Fetching the three requirements.
        """
        pv_simple = sum([roof.get_pv_simple() for roof in self.roofs])  # in kW
        pv_standard_area = sum([roof.get_pv_standard() for roof in self.roofs])  # in m²
        pv_advanced_area = sum([roof.get_pv_advanced() for roof in self.roofs])  # in m²
        """
        Translating m² to kWp
        """
        pv_standard = pv_standard_area*PV_DEF_EFF_FACTOR
        pv_advanced = pv_advanced_area*PV_DEF_EFF_FACTOR
        """
        Finding the minimum requirement.
        """
        pv_need = np.min([pv_simple, pv_standard, pv_advanced]) - solar_thermal_pv_equivalent
        return pv_need  # in kWp


    # def get_yearly_heating_el_factor(self):
    #     factor = sum(heating_system. for heating_circuit in self.heating_circuits for heating_system in heating_circuit.heating_systems)
    #     return 0

    @staticmethod
    def _categorize_years(year, thresholds, values):
        """Categorizes years into bins based on thresholds and returns corresponding values."""
        
        # Ensure NumPy arrays
        thresholds = np.asarray(thresholds)
        values = np.asarray(values)

        # Use searchsorted to find the index of the category the year belongs to
        category_index = np.searchsorted(thresholds, year, side='right')

        # Return the corresponding value (handle out-of-bounds years)
        if 0 <= category_index < len(values):
            return values[category_index]
        else:
            return None  # Or a default value for years outside the thresholds
        
    def set_ht(self) -> None:
        """
        Sets the specific heat loss thingy in W/m²K.
        """
        if self.ht is not None:
            return
        
        years_threshholds = THRESHOLD_YEARS_HEAT_LOSS  # from typology tables for buildings in enev 2015
        values = VALUES_HEAT_LOSS

        self.ht = Building._categorize_years(self.con_year, years_threshholds, values)
    
    def set_volume(self) -> None:
        """
        Sets the air and total volume in m³ for the building based on the living area or air volume given.
        """
        if not (self.air_volume is None or self.air_volume == 0):
            self.total_volume = DEF_FAC_AIR_TOT_VOL*self.air_volume
            return
        
        self.air_volume = self.liv_area*DEFAULT_HEIGHT
        self.total_volume = DEF_FAC_AIR_TOT_VOL*self.air_volume

    def set_air_exchange_rate(self) -> None:
        """
        Sets the air exchange rate in total for window circulation and infiltration, at normal conditions in 1/h for the building based on the year it was built
        """
        if not (self.air_exchange_rate is None or self.air_exchange_rate == 0):
            return
        
        threshold_years = THRESHOLD_YEARS_AIR_EXCHANGE
        values = VALUES_AIR_EXCHANGE

        self.air_exchange_rate = Building._categorize_years(self.con_year, threshold_years, values)

    def set_hull_area(self) -> None:
        """
        Sets the hull area in m² for the building based on the year it was built
        """
        if not (self.hull_area is None or self.hull_area == 0):
            return
        if not (self.total_volume is None or self.total_volume == 0):
            self.set_volume()

        self.hull_area = self.total_volume*DEFAULT_A_TO_VE

    def get_yearly_heating_demand(self):
        "Returns the total energy demand for heating the building the entire year, excluding warm water demand."
        self.set_ht()
        self.set_volume()
        self.set_air_exchange_rate()
        return 

    def get_yearly_base_el_demand(self) -> float:
        """
        Returns the yearly base electricity demand in kWh not including heating, cooling, air conditioning, or electric cars. If it already exists, it simply returns the value. 
        If it does not, we try to calculate it. 
        If not enough parameters to calculate, we send the user back to their fantasy land.
        If parameters can't be added or multiplied, the user is also trying to fuck with us. So he gets an error message.
        """
        if self.base_el_consumption is not None:
            return self.base_el_consumption
        if self.liv_area is None or self.residents is None or self.large_el_devices is None:
            raise ValueError("Not enough parameters")
        try:
            self.base_el_consumption = float(self.liv_area*FACTOR_LIV_AREA + self.residents*FACTOR_RESIDENTS + self.large_el_devices*FACTOR_BIG_EL_DEVICES)
            return self.base_el_consumption
        except (TypeError, ValueError):
            raise ValueError("We need numbers for the values. Not whatever the hell you put in! (living area, number of residents, or number of electrical appliances is not a number)")

    def set_max_heat_demand(self) -> None:
        """
        sets the max heat demand in kW.
        """
        if not (self.max_heat_demand is None or self.max_heat_demand == 0):
            return
        self.set_ht()
        self.set_volume()
        self.set_air_exchange_rate()
        self.set_hull_area()

        temp_dif = self.standard_room_temp - self.outside_air_temp

        heat_demand_transmission = self.ht*self.hull_area*temp_dif/1000  # in kW
        heat_demand_air_exchange = self.air_exchange_rate*self.air_volume*AIR_DENSITY*AIR_HEAT_CONSTANT*temp_dif/3600/1000  # om kW

        heat_demand =  heat_demand_transmission + heat_demand_air_exchange  # in kW
        self.max_heat_demand = heat_demand

    def get_max_heat_demand(self) -> float:
        """
        return: the max heat demand of the building for a norm winter day based on the assumptions. Result in kW.
        """
        self.set_max_heat_demand()
        return self.max_heat_demand
    
    def set_warm_water_demand(self) -> None:
        """
        sets the max heat demand for warm drinking water of the building based on the assumptions. Result in kW.
        """
        if not (self.warm_water_demand is None or self.warm_water_demand == 0):
            return
        self.warm_water_demand = self.residents*WARM_WATER_FACTOR/1000

    def get_warm_water_demand(self):
        """
        return: the max heat demand for warm drinking water of the building based on the assumptions. Result in kW.
        """
        self.set_warm_water_demand()
        return self.warm_water_demand

    def set_total_help_energy(self) -> None:
        """
        sets the energy demand for the pumps and automation of the building based on the assumptions. Result in kW. Is always supplied by electricity
        """
        if not (self.help_energy is None or self.help_energy == 0):
            return
        heat_systems = [heat_system for heating_circuit in self.heating_circuits for heat_system in heating_circuit.heating_systems]
        help_energy = sum([heat_system.help_electricity for heat_system in heat_systems])
        self.help_energy = help_energy

    def get_total_help_energy(self):
        """
        return: the energy demand for the pumps and automation of the building based on the assumptions. Result in kW. Is always supplied by electricity
        """
        self.set_total_help_energy()
        return self.help_energy

    def set_electricity_share_heating(self) -> None:
        """
        sets the energy demand for the pumps and automation of the building based on the assumptions. Result in kW. Is always supplied by electricity
        """
        if not (self.electricity_share_heating is None or self.electricity_share_heating == 0):
            return
        electricity_share_heating = 0
        for heating_circuit in self.heating_circuits:
            circuit_share_in_building = heating_circuit.share  # the share of the circuit in the building. If only one circuit, it should be 1.
            for heating_system in heating_circuit.heating_systems:
                system_share_in_circuit = heating_system.share  # share of the heating system in the circuit. If only one heating system, it should be 1.
                system_share_in_building = system_share_in_circuit*circuit_share_in_building  # share of system in building.

                if heating_system.eng_source_ind == EL_ENG_IND:  # which is electricity
                    electricity_share_heating += system_share_in_building  # if the system is electric, we add the share to the electric share.

        self.electricity_share_heating = electricity_share_heating

    def get_electricity_share_heating(self):
        """
        return: the energy demand for the pumps and automation of the building based on the assumptions. Result in kW. Is always supplied by electricity
        """
        self.set_electricity_share_heating()
        return self.electricity_share_heating

    def get_pv_max_power(self):
        if self.pv_system is None:
            return 0
        return sum([pv.max_power for pv in self.pv_system])

    def get_pv_generation_year(self) -> np.ndarray:
        pv_pow = self.get_pv_max_power()

        if pv_pow == 0:
            return 0
        
        generation = np.sum([pv.get_pv_generation_year() for pv in self.pv_system], axis=0)
        return generation
        
    def set_average_el_heat_eff(self) -> None:
        """
        sets the average efficiency of the electric heating systems.
        """
        if not (self.average_el_heat_eff is None or self.average_el_heat_eff == 0):
            return
        
        heat_produced = 0
        elec_needed = 0
        for heating_circuit in self.heating_circuits:
            circuit_share_in_building = heating_circuit.share  # the share of the circuit in the building. If only one circuit, it should be 1.
            for heating_system in heating_circuit.heating_systems:
                if heating_system.eng_source_ind == EL_ENG_IND:  # which is electricity
                    system_share_in_circuit = heating_system.share  # share of the heating system in the circuit. If only one heating system, it should be 1.
                    system_share_in_building = system_share_in_circuit*circuit_share_in_building
                    system_eff = heating_system.get_efficiency(flow_temp=heating_circuit.flow_temp, ret_temp=heating_circuit.ret_temp)
                    heat_produced +=  system_share_in_building # if the system is electric, we add the specific heat produced.
                    elec_needed += system_share_in_building/system_eff  # adding the specific electricity needed. 
                    # It can all be multiplied by the total heat demand of the building to get the real values in kWh.

        self.average_el_heat_eff = heat_produced/elec_needed

    def get_average_el_heat_eff(self):
        """
        return: the average efficiency of the electric heating systems.
        """
        self.set_average_el_heat_eff()
        return self.average_el_heat_eff
       
    # def get_subsidy(self):
    #     cur_year = datetime.datetime.now().year
    #     c_year = self.heating.con_year
    #     h_cost = self.heating.cost
    #     if c_year != self.old_heating.con_year and c_year >= cur_year-1:
    #         base_sub = self.heating.get_base_subsidy()
    #         old_eng_src = self.old_heating.get_eng_source()
    #
    #         if old_eng_src == "Oil" or old_eng_src == "Electricity" or (old_eng_src == "Gas"
    #         and c_year <= cur_year - 20):
    #             return (base_sub + 10)/100*h_cost
    #         return base_sub/100*h_cost
    #     return 0

    # def __str__(self):
    #     return f"Home with the construction year: {self.con_year}\nand the heating system: {self.heating}"


class EnergyManagement:

    def __init__(self, homes: list[Building], grid_oriented:bool = False, complex_method:bool = False) -> None:
        """
        Class that calculates the energy and economics of the system of buildings.
        homes: list of the homes that are a part of this pv sharing system.
        grid_oriented: when True, we simply spread out the load of the electric vehicles in case of smart meters. 
        When false we try to use as much of our own PV generation to save money.

        """
        self.homes = homes
        self.grid_oriented = grid_oriented
        self.complex_method = complex_method

        self.pv_winter = None
        self.pv_summer = None
        self.current_car = None
        self.base_yearly_demand = None
        self.yearly_el_car_demand = None
        self.total_demand = None
        self.pv_generation = None
        
        self.max_heat_temp = MAX_TEMP_HEATING
        self.temp_dif = STANDARD_ROOM_TEMP - STANDARD_OUTSIDE_TEMP

        self.max_heat_demand = None
        self.warm_water_demand = None
        self.help_energy = None
        self.electricity_share_heating = None
        self.average_electric_eff = None
    
    def reset_vars(self, homes: list[Building] = None, grid_oriented:bool = None, complex_method:bool = None):
        self.pv_winter = None
        self.pv_summer = None
        self.current_car = None
        self.base_yearly_demand = None
        self.yearly_el_car_demand = None
        self.total_demand = None
        self.pv_generation = None
        
        self.max_heat_temp = MAX_TEMP_HEATING
        self.temp_dif = STANDARD_ROOM_TEMP - STANDARD_OUTSIDE_TEMP

        self.max_heat_demand = None
        self.warm_water_demand = None
        self.help_energy = None
        self.electricity_share_heating = None
        self.average_electric_eff = None

        self.homes = homes if homes is not None else self.homes
        self.grid_oriented = grid_oriented if grid_oriented is not None else self.grid_oriented
        self.complex_method = complex_method if complex_method is not None else self.complex_method

    def are_all_smart(self):
        """
        Returns if every building in this system has a smart meter installed.
        """
        is_smart = np.all([building.smartmeter for building in self.homes])
        return is_smart

    def simulate_year(self, year:int = DEFAULT_YEAR):
        """
        Simulates the year. Checks if battery is existent or not and uses correct method.
        """
        battery_cap = self.get_battery_cap()

        if battery_cap == 0:  # case no batteries available.
            return self._simulate_year_no_battery()
        
        # if there is a battery, we do the other method.
        return self._simulate_year_with_battery()
    
    def _simulate_year_no_battery(self):
        """
        Simulates the year if buildings have no batteries.
        """
        total_el_demand = self.get_total_electric_yearly_demand()
        pv_values_year = self.get_pv_generation_year()

        self_use = np.min([total_el_demand, pv_values_year], axis=0)
        feed_in = pv_values_year - self_use
        from_grid = total_el_demand - self_use

        """
        Create index
        """

        year = DEFAULT_YEAR
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))

        dt_index = pd.Index(times, name=TIMESTAMP_STRING)
        """
        Get amounts
        """
        self_use_amount = np.sum(self_use)/6  # divide by six because of six hourly values. Interval is ten minutes.
        feed_in_amount = np.sum(feed_in)/6
        from_grid_amount = np.sum(from_grid)/6

        data = np.array([self_use, feed_in, from_grid]).T
        df = pd.DataFrame(data=data, index=dt_index, columns=[SELF_USE_STRING, FEED_IN_STRING, FROM_GRID_STRING])

        ret_str = str(f"self use amount was {round(self_use_amount, 2)} kWh, the feed in amount was "
                      f"{round(feed_in_amount, 2)} kWh, and the amount needed from the grid was {round(from_grid_amount, 2)} kWh")
        
        # return self_use_amount, feed_in_amount, from_grid_amount
        return df

    def _simulate_year_with_battery(self):
        """
        Simulates the year if battery exists.
        """
        total_el_demand = self.get_total_electric_yearly_demand()
        pv_values_year = self.get_pv_generation_year()
        battery_cap = self.get_battery_cap()

        net_generation = pv_values_year - total_el_demand
        last_soc = 0
        net_battery_exp = np.zeros(len(net_generation))  # initializing the self use array.
        possible_battery_charge = np.clip(net_generation, - DEFAULT_BATTERY_MAX_DISCHARGE*battery_cap, DEFAULT_BATTERY_MAX_CHARGE*battery_cap)
        # the battery has maximums for charging and discharging and this accounts for that.

        for i in range(len(possible_battery_charge)):  # only need to calculate the net_battery_exp part in the iterative way. The rest can be done vectorially.
            cur_soc = np.clip(last_soc + possible_battery_charge[i], 0, battery_cap)  # current stand of the battery is dependent on net generation and physical limits. (empty, full).
            net_battery_exp[i] = last_soc - cur_soc  # the net battery export. Could be defined as net import and have the opposite sign but this is my convention.
            last_soc = cur_soc
        
        post_losses_net_battery_exp = (net_battery_exp < 0)*DEFAULT_BATTERY_EFF_IN*net_battery_exp + (net_battery_exp > 0)*DEFAULT_BATTERY_EFF_OUT*net_battery_exp    
        # getting the efficiency of the battery to do it's thing.

        bat_losses = np.abs(net_battery_exp) - np.abs(post_losses_net_battery_exp)  # the losses of the battery.

        self_use = np.min([total_el_demand, pv_values_year + net_battery_exp - bat_losses], axis=0)  # the max amount that can be self consumed is the current demand. 
        # It also can't be higher than the sum of the generation and net battery export. Even if the battery export is negative, this should work. I think...

        feed_in = pv_values_year + post_losses_net_battery_exp - self_use  # we feed in our total generation minus what we use ourselves.
        # if 
        from_grid = total_el_demand - self_use

        """
        Create index
        """
        year = DEFAULT_YEAR
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))

        dt_index = pd.Index(times, name=TIMESTAMP_STRING)
        """
        Get amounts
        """

        self_use_amount = np.sum(self_use)/6  # divide by six because of six hourly values. Interval is ten minutes.
        feed_in_amount = np.sum(feed_in)/6
        from_grid_amount = np.sum(from_grid)/6

        ret_str = str(f"self use amount was {round(self_use_amount, 2)} kWh, the feed in amount was "
                      f"{round(feed_in_amount, 2)} kWh, and the amount needed from the grid was {round(from_grid_amount, 2)} kWh")
        
        """
        Create df
        """
        data = np.array([self_use, feed_in, from_grid]).T

        df = pd.DataFrame(data=data, index=dt_index, columns=[SELF_USE_STRING, FEED_IN_STRING, FROM_GRID_STRING])

        return df
        # return self_use_amount, feed_in_amount, from_grid_amount

    def get_total_electric_yearly_demand(self) -> np.ndarray:
        yearly_electric_heating_demand = self.get_yearly_electric_heating_demand()
        yearly_el_car_demand = self.get_yearly_el_car_demand()
        yearly_el_car_demand = yearly_el_car_demand[ELDEMAND_STRING].to_numpy()
        yearly_base_demand = self.get_yearly_base_demand()

        total_el_demand = np.sum([yearly_base_demand, yearly_el_car_demand, yearly_electric_heating_demand], axis=0)
        return total_el_demand

    def get_yearly_base_demand(self) -> np.ndarray:
        """
        returns the yearly base demand in kW, in the same intervals as in the database (currently 10 min), adjusted for the consumption of the building.
        """
        df_el_demand = get_el_demand_dataframe()

        from_third_column = df_el_demand.iloc[:, 2:]

        el_demands = from_third_column.values.flatten(order="C")  # since the columns are the times 00:00 00:10 ... we need to flatten it.

        yearly_el_consumption = self.get_yearly_base_el_consumption()  # the consumption of the system of buildings in kWh.

        return el_demands*(yearly_el_consumption/DEFAULT_EL_DEMAND/1000)  # divide by DEFAULT_EL_DEMAND because the database has DEFAULT_EL_DEMAND as the base demand.
        # then divide by 1000 to get from W to kW.
    
    def get_yearly_base_el_consumption(self) -> float:
        """
        return: the yearly electric consumption for non-heating or electric vehicle use in kWh.
        """
        return sum([building.get_yearly_base_el_demand() for building in self.homes])

    def get_pv_generation_year(self) -> np.ndarray:
        """
        Returns the ten minute values of pv generation for this system. There's a lot behind this little dawg. Or rather the set method it calls.
        """
        self.set_pv_generation_year()
        return self.pv_generation
    
    def set_pv_generation_year(self) -> np.ndarray:
        """
        Sets the yearly pv generation array of ten minute values.
        """
        self.pv_generation = np.sum([home.get_pv_generation_year() for home in self.homes], axis=0)
        return

    def get_battery_cap(self) -> float:
        bat_cap = sum([home.battery_cap for home in self.homes])
        bat_cap = bat_cap*DEFAULT_BATTERY_YEARLY_DEGRADATION**DEFAULT_BATTERY_AGE
        return bat_cap

    def get_yearly_electric_heating_demand(self, year:int = DEFAULT_YEAR) -> np.ndarray:
        """
        return: the heating demand covered by electricity over the entire year in the same intervals as in the database. (ten minutes as of July 2024)
        """
        help_eng = self.get_help_energy()
        ww_demand = self.get_warm_water_demand()
        max_heat_demand = self.get_max_heat_demand()
        df_sol = get_solar_db()
        electricity_share_heating = self.get_electricity_share_heating()
        electric_average_efficiency = self._get_average_electric_eff()

        outside_temp = df_sol[TEMP_AIR_STRING].to_numpy()
        heating_demand = (outside_temp < 15)*(STANDARD_ROOM_TEMP-outside_temp)/(self.temp_dif)*max_heat_demand  # calculating the heating demand for the year.
        heating_demand = heating_demand + ww_demand  # adding the constant warm water demand.

        eff_factors = np.interp(outside_temp, HEAT_PUMP_OUT_TEMP_REF_TEMPS, HEAT_PUMP_OUT_TEMP_FACTORS)
        efficiencies = eff_factors*electric_average_efficiency
        
        electric_demand_heating = heating_demand*electricity_share_heating  # accounting for the fact that not all generators are electric based.
        electric_demand_heating = electric_demand_heating/efficiencies  # dividing here because the generators efficiency 
        # affects the warm water demand and heat demand but not the auxiliary demand by the pumps and regulation and all that.

        electric_demand_heating = electric_demand_heating + help_eng  # adding the energy needed for pumps and regulation since that is always electric based.
        
        return electric_demand_heating
    
    def get_yearly_el_car_demand(self, year:int = DEFAULT_YEAR):
        """
        return: the electricity demand over the entire year in the same intervals as in the database. (ten minutes as of July 2024)
        """
        self.set_yearly_elcar_demand()
        return self.yearly_el_car_demand
    
    def set_yearly_elcar_demand(self, year:int = DEFAULT_YEAR) -> None:
        """
        Returns: Nothing.
        Sets the yearly demand for all electric cars in this system.
        Does not work if some parameters have changed but a current yearly demand was already created.
        Once one demand table has been created, it is impossible to create another with this function.
        """
        if self.yearly_el_car_demand is not None:
            return
        smarts = [home.smartmeter for home in self.homes]
        smart = np.all(smarts)
        bat_cap = self.get_battery_cap()
        have_pv = sum([home.get_pv_max_power() for home in self.homes]) > 0
        if (not (smart and have_pv)) or self.grid_oriented:
            self.yearly_el_car_demand = self._get_yearly_simple_el_cars_demand(year, smart)
            return
        if self.complex_method:
            if bat_cap > 0:
                self.yearly_el_car_demand = self._get_yearly_genius_with_battery()
                return
            self.yearly_el_car_demand = self._get_yearly_genius_no_battery()
            return
        self.yearly_el_car_demand = self._get_yearly_simple_smart_demand(year)
    
    def set_help_energy(self) -> None:
        """
        Sets the help energy of the system.
        """
        if not (self.help_energy is None or self.help_energy == 0):
            return
        
        buildings = [building for building in self.homes]
        help_energies = [building.get_total_help_energy() for building in buildings]
        self.help_energy = sum(help_energies)
    
    def get_help_energy(self) -> float:
        """
        return: The help energy of the system in kW.
        """
        self.set_help_energy()
        return self.help_energy

    def set_max_heat_demand(self) -> None:
        """
        Sets the max heat demand of the system.
        """
        if not (self.max_heat_demand is None or self.max_heat_demand == 0):
            return
        
        buildings = [building for building in self.homes]
        max_heat_demands = [building.get_max_heat_demand() for building in buildings]
        self.max_heat_demand = sum(max_heat_demands)

    def get_max_heat_demand(self):
        """
        return: The max heat demand of the system in kW.
        """
        self.set_max_heat_demand()
        return self.max_heat_demand

    def set_warm_water_demand(self) -> None:
        """
        Sets the warm water demand of the system in kW.
        """
        if not (self.warm_water_demand is None or self.warm_water_demand == 0):
            return
        
        buildings = [building for building in self.homes]
        warm_water_demands = [building.get_warm_water_demand() for building in buildings]
        self.warm_water_demand = sum(warm_water_demands)

    def get_warm_water_demand(self):
        """
        return: the warm water demand of the system in kW.
        """
        self.set_warm_water_demand()
        return self.warm_water_demand

    def set_electricity_share_heating(self) -> None:
        """
        Sets the electricity share of the system in the heating. goes from 0 to 1.
        """
        if not (self.electricity_share_heating is None or self.electricity_share_heating == 0):
            return
        
        buildings = [building for building in self.homes]
        max_heat_demand = self.get_max_heat_demand()  # max demand for entire system.

        buildings_max_demands = np.array([building.get_max_heat_demand() for building in buildings], dtype=float)  # list of max demands for each building
        buildings_share_in_demand = buildings_max_demands/max_heat_demand  # shares of max demand for each building. Buildings that require more heat will have a higher share.

        electricity_shares_heating = np.array([building.get_electricity_share_heating() for building in buildings])  # the shares of electricity demand for each building.

        real_shares = electricity_shares_heating*buildings_share_in_demand  # weighting the electricity demands by building demand.
        self.electricity_share_heating = sum(real_shares)  # sum of singular weighted shares for buildings should be the share for the entire system. 

    def get_electricity_share_heating(self):
        """
        return: the electricity share of the system in the heating. goes from 0 to 1.
        """
        self.set_electricity_share_heating()
        return self.electricity_share_heating

    def _set_average_electric_eff(self) -> None:
        """
        Sets the average efficiency of the electric heating systems.
        """
        if not (self.average_electric_eff is None or self.average_electric_eff == 0):
            return
                
        buildings = [building for building in self.homes]
        max_heat_demand = self.get_max_heat_demand()  # max demand for entire system.

        buildings_max_demands = np.array([building.get_max_heat_demand() for building in buildings], dtype=float)  # list of max demands for each building
        buildings_share_in_demand = buildings_max_demands/max_heat_demand  # shares of max demand for each building. Buildings that require more heat will have a higher share.

        electricity_shares_heating = np.array([building.get_electricity_share_heating() for building in buildings])  # the shares of electricity demand for each building.

        real_shares = electricity_shares_heating*buildings_share_in_demand  # weighting the electricity demands by building demand.
        self.electricity_share_heating = sum(real_shares)  # sum of singular weighted shares for buildings should be the share for the entire system. 

        buildings_el_eff = np.array([building.get_average_el_heat_eff() for building in buildings])
        heat_produced = real_shares  # the array of real electricity shares is also the array for the specific heat produced by electric generators.
        electricity_needed = heat_produced/buildings_el_eff  # dividing the specific heat produced by the electric 
        # generators of a building by the average efficiency of said generators should give back the specific amount of electricity needed.
        # multiply by total demand to get real values in kWh.

        total_heat_produced = np.sum(heat_produced)
        total_electricity_needed = np.sum(electricity_needed)

        self.average_electric_eff = total_heat_produced/total_electricity_needed

    def _get_average_electric_eff(self):
        """
        return: the average efficiency of the electric heating systems.
        """
        self._set_average_electric_eff()
        return self.average_electric_eff
      
    def _get_yearly_simple_el_cars_demand(self, year, smart: bool = False):
        """
        :return: the full cars charging schedule for the case of smart meter, but no pv, cause this household is full of cowards.
        """
        db_week = self._get_db_weekly_all_cars(smart=smart)
        db_week_vals = db_week.loc[:, ELDEMAND_STRING].to_numpy()
        db_year = self.get_yearly_demand_from_weekly_demand(db_week_vals, year)
        return db_year
    
    def _get_db_weekly_all_cars(self, smart: bool = False):
        """
        For all cars unlike function above.
        Creates a weekly schedule of electric charging and returns the dataframe. Using 2024 because it started on a monday.
        The year doesn't get saved later and plays no part in the returned database. 
        The grid has a connected energy system with the heating devices and PV power being shared.
        :return: Dataframe of weekly schedule of electric charging for every car in every building in this smart grid.
        """

        """
        Initializing shit.
        """
        date_start = np.datetime64(f"{2024}-01-01T00:00")  # 2024 cause it started on Monday
        date_stop = np.datetime64(f"{2024}-01-08T00:00")  # week later unless my math is off.
        deltie = np.timedelta64(10, "m")  # 10 minutes my dawg.

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times_week = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))  # times for the week with 10 minutes as step

        demands_week = np.zeros(times_week.shape[0])  # assume no demand until proven otherwise.
        
        # indexes_car_temp = [[timey.day-1, timey.time()] for timey in times_week]  # the dataframe has the indexes day of week and time. 
        # index_week = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)

        index_week = pd.Index(times_week, name=TIMESTAMP_STRING)

        db_week = pd.DataFrame(columns=[ELDEMAND_STRING], data=demands_week, index=index_week)  # dataframe with correct index and column name but full of 0s.

        cars = self.get_all_el_cars()

        for _, car in enumerate(cars):  # since our "get_specific_weekly_stupid" method works for each car at a time, we must iterate over every car.
            if smart:
                updater = self.get_specific_week_smartnopv(car)
            else:
                updater = self.get_specific_weekly_stupid(car)
            db_week = db_week.add(updater, fill_value= 0)  # it doesn't work right. I don't know why.

        return db_week

    @staticmethod
    def get_specific_week_smartnopv(e_car:ElCar):
        """
        :param e_car: The electric car.
        return: Dataframe of The added charge demand caused by this car for the case of smart meter but no PV. This is for one car only. 
        The function below is for all cars. Confusing shit, I know.
        """
        rest_energy = 0
        values_car_temp = np.empty(shape=0)
        indexes_car_temp = np.empty(shape=0, dtype=np.dtypes.DateTime64DType)
        delt_10_min = np.timedelta64(10, "m")

        for indiana, cycle in enumerate(e_car.el_schedule):
            next_cycle = get_next_element(indiana, e_car.el_schedule)
            charge_time = get_charge_time(cycle,next_cycle)
            weekday_back, time_back = cycle.get_time_back()

            np_charge_time_rounded = np.timedelta64(int(round(charge_time*6, 0)*10), "m")  # We round it because the database is in intervals of 10 minutes.
            energy_need = min(cycle.distance/e_car.efficiency + rest_energy, e_car.battery_cap)  # if the car goes below zero on fuel, 

            demand = min(e_car.max_charge_pow, energy_need/charge_time)
            rest_energy = energy_need - demand * charge_time  # if not enough time to charge fully, the battery is left partly empty
            
            start_datetime = np.datetime64(f"2024-01-0{weekday_back+1}T{time_back}")  # starting to charge directly after getting back from the current trip.
            end_datetime = start_datetime + np_charge_time_rounded  # we assume time_back is a valid time and already rounded to the nearest 10 minutes.

            new_inds = np.arange(start=start_datetime, stop=end_datetime, step=delt_10_min)  # these should be all the timestapmps where charging occurs.
            indexes_car_temp = np.append(indexes_car_temp, new_inds)  # adding our new timestamps into the indexes array.
            index_len = len(new_inds)  # could just be one line with the next, but it would be a bit cluttered.
            values_car_temp =  np.append(values_car_temp, np.ones(shape=index_len, dtype=float)*demand)  # the demand is constant and therefore can be applied to every timestamp in the indexes.

        
        """
        Pushing the dates that wrap around the week back to their rightful place.
        """
        threshold_date = pd.to_datetime('2024-01-08')
        indexes_car_temp = np.array(indexes_car_temp, dtype=np.dtypes.DateTime64DType)
        start_index = np.searchsorted(indexes_car_temp, threshold_date)  # I*ll be honest with you, I don't remember how this works.
        if start_index < len(indexes_car_temp)-1:  # no clue.
            week_offset = np.timedelta64(7, "D").astype('datetime64[D]')
            indexes_car_temp[start_index:] = indexes_car_temp[start_index:] - week_offset  # It tries to push back anything past- 
            # a week to the start of the month, but does it work? no idea.

        """
        creating the dataframe.
        """
        # indexes_car_temp = [[d.day-1, d.time()] for d in indexes_car_temp]  # the dataframe has the indexes day of week and time. 
        # # 2024 starts on monday, so we push back the day of month by one and get the day of the week.

        # index = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)
        
        index = pd.Index(indexes_car_temp, name=TIMESTAMP_STRING)

        db_week_car_update = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=values_car_temp)  # we use update because- 
        # otherwise the database should be full of zeros when the car isn't charging.
        return db_week_car_update

    @staticmethod
    def get_specific_weekly_stupid(e_car:ElCar) -> pd.DataFrame:
        """
        :param e_car: The electric car.
        return: Dataframe of The added charge demand caused by this car for the case of no smart meter and no PV. 
        We're both stupid and cheap here. PV is for hippies and being smart is for nerds. This is for one car only. 
        """
        rest_energy = 0
        values_car_temp = np.empty(shape=0)
        indexes_car_temp = np.empty(shape=0, dtype=np.dtypes.DateTime64DType)
        delt_10_min = np.timedelta64(10, "m")

        for indiana, cycle in enumerate(e_car.el_schedule):
            next_cycle = get_next_element(indiana, e_car.el_schedule)
            max_charge_time = get_charge_time(cycle, next_cycle)
            weekday_back, time_back = cycle.get_time_back()

            energy_need = min(cycle.distance/e_car.efficiency + rest_energy, e_car.battery_cap)  # if the car goes below zero on fuel, 
            # we assume they just charged somwehere else and came back empty. Battery level can't be negative.

            demand = e_car.max_charge_pow  # we have no smart meters or other way to control the charging speed. 
            # We're just stupidly sending electrons to their death. So it's just the max possible for the car.
            # In reality, maybe the charging station doesn't support the max charging speed of the car. But that's an issue for another day.
            # I've wasted long enough on this project.

            charge_time = min(max_charge_time, energy_need/demand)  # Either we charge to full, 
            # or we charge until we can't charge anymore because the owner is going on another trip before the battery is full.
            
            
            np_charge_time_rounded = np.timedelta64(int(round(charge_time*6, 0)*10), "m")  # We round it because the database is in intervals of 10 minutes.

            rest_energy = energy_need - demand * charge_time  # if not enough time to charge fully, the battery is left partly empty
            start_datetime = np.datetime64(f"2024-01-0{weekday_back+1}T{time_back}")  # starting to charge directly after getting back from the current trip.
            end_datetime = start_datetime + np_charge_time_rounded  # we assume time_back is a valid time and already rounded to the nearest 10 minutes.

            new_inds = np.arange(start=start_datetime, stop=end_datetime, step=delt_10_min, dtype=np.dtypes.DateTime64DType)  # these should be all the timestapmps where charging occurs.
            indexes_car_temp = np.append(indexes_car_temp, new_inds)  # adding our new timestamps into the indexes array.
            index_len = len(new_inds)  # could just be one line with the next, but it would be a bit cluttered.
            demand_array = np.ones(shape=index_len, dtype=float)*demand
            demand_array[-1] -= (sum(demand_array)/6 -energy_need)*6
            values_car_temp =  np.append(values_car_temp, demand_array)  # the demand is constant and therefore can be applied to every timestamp in the indexes.
        
        """
        Pushing the dates that wrap around the week back to their rightful place.
        """
        indexes_car_temp = np.array(indexes_car_temp, dtype=np.dtypes.DateTime64DType)
        threshold_date = pd.to_datetime('2024-01-08')

        start_index = np.searchsorted(indexes_car_temp, threshold_date)  # I*ll be honest with you, I don't remember how this works.
        if start_index < len(indexes_car_temp)-1:  # no clue.
            week_offset = np.timedelta64(7, "D").astype('datetime64[D]')
            indexes_car_temp[start_index:] = indexes_car_temp[start_index:] - week_offset  # It tries to push back anything past- 
            # a week to the start of the month, but does it work? no idea.

        """
        creating the dataframe.
        """
        # indexes_car_temp = [[d.day-1, d.time()] for d in indexes_car_temp]  # the dataframe has the indexes day of week and time. 
        # # 2024 starts on monday, so we push back the day of month by one and get the day of the week.

        # index = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)
        
        index = pd.Index(indexes_car_temp, name=TIMESTAMP_STRING)

        db_week_car_update = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=values_car_temp)  # we use update because- 
        # otherwise the database should be full of zeros when the car isn't charging.
        return db_week_car_update
    
    def _get_yearly_simple_smart_demand(self, year):
        """
            A function for the summer demand in case of smart meters in the simple method of prioritizing times between
            07:00 and 16:00. In this method the charging schedule is exact for every week.

            :param year: the year you want printed.
            :return: The summer charging schedule of the electric car, prioritizing times with solar radiation.
        """

        """
        Initializing usual stuff.
        """
        self.initiate_pv_sum_win()
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))

        """
        Initializing weekly stuff. including the summer and winter schedules.
        """
        db_week_summer = self._get_db_week_simsmart_season(is_summer=True)
        db_week_winter = self._get_db_week_simsmart_season(is_summer=False)

        db_week_summer_vals = db_week_summer.loc[:, ELDEMAND_STRING].to_numpy()
        db_week_winter_vals = db_week_winter.loc[:, ELDEMAND_STRING].to_numpy()
        len_db_weekvals = len(db_week_summer_vals)

        """
        initializing yearly stuff.
        """
        datalength = len(times)
        data = np.empty(dtype=float, shape=datalength)

        """
        Getting the cutofffs from the default global ones. The values are as of July 3rd 2024: 91 and 274.
        """
        cutoff_1 = int(DAYS_TRANSITION_1*datalength/365)
        cutoff_2 = int(DAYS_TRANSITION_2*datalength/365)
        """
        creating the needed variables for the 1st winter period.
        """
        weekday_start_Winter1 = pd.to_datetime(date_start).weekday()
        start_ind_weekly_Winter1 = int(len_db_weekvals/7*weekday_start_Winter1)  # If we start on monday, this is zero. 
        # If we start on tuesday this is 24*6=144 if the intervals are ten minutes. 
        end_index_first_week_Winter1 = len_db_weekvals - start_ind_weekly_Winter1  # This is the index for the end of week 1 of the year. 
        # It should be the length of a week minus the steps we skip.
        rest_datalength_Winter1 = cutoff_1 - end_index_first_week_Winter1
        repeat_count_Winter1 = int(rest_datalength_Winter1 // len_db_weekvals )
        remainder_Winter1 = int(rest_datalength_Winter1 % len_db_weekvals)

        """
        creating the needed variables for the summer period.
        """
        weekday_start_summer = int((weekday_start_Winter1 + DAYS_TRANSITION_1)%7)  # if I'm not stupid (which I definitely am), 
        # this should work. Since weeks are kind of a cycle. I just need to know how many days forward we're jumping.
        start_ind_weekly_Summer = int(len_db_weekvals/7*weekday_start_summer)  # If we start on monday, this is zero. 
        # If we start on tuesday this is 24*6=144 if the intervals are ten minutes. 
        end_index_first_week_Summer = cutoff_1 + len_db_weekvals - start_ind_weekly_Summer  # This is the index for the end of week 1 of the year. 
        # It should be the length of a week minus the steps we skip.
        rest_datalength_Summer = cutoff_2 - end_index_first_week_Summer
        repeat_count_Summer = int(rest_datalength_Summer // len_db_weekvals )
        remainder_Summer = int(rest_datalength_Summer % len_db_weekvals)

        """
        creating the needed variables for the 2nd winter period.
        """
        weekday_start_Winter2 = int((weekday_start_Winter1 + DAYS_TRANSITION_2)%7)  # if I'm not stupid (which I definitely am), 
        # this should work. Since weeks are kind of a cycle. I just need to know how many days forward we're jumping.
        start_ind_weekly_Winter2 = int(len_db_weekvals/7*weekday_start_Winter2)  # If we start on monday, this is zero. 
        # If we start on tuesday this is 24*6=144 if the intervals are ten minutes. 
        end_index_first_week_Winter2 = cutoff_2 + len_db_weekvals - start_ind_weekly_Winter2  # This is the index for the end of week 1 of the year. 
        # It should be the length of a week minus the steps we skip.
        rest_datalength_Winter2 = datalength - end_index_first_week_Winter2
        repeat_count_Winter2 = int(rest_datalength_Winter2 // len_db_weekvals )
        remainder_Winter2 = int(rest_datalength_Winter2 % len_db_weekvals)

        """
        Adding the weekly values to the data. Starting with the first winter period.
        """
        data[:end_index_first_week_Winter1] = db_week_winter_vals[start_ind_weekly_Winter1:]  # the first weekday of the period doesn't have to be monday-
        # if the start is on another day, this adjusts for that.
        data[end_index_first_week_Winter1:end_index_first_week_Winter1 + repeat_count_Winter1 * len_db_weekvals] = np.tile(db_week_winter_vals, repeat_count_Winter1)
        # this part is for most of the weeks. every full week in the period gets repeated.
        data[end_index_first_week_Winter1 + repeat_count_Winter1 * len_db_weekvals:cutoff_1] = db_week_winter_vals[:remainder_Winter1] 
        # this part is for the last week of the period. It can end before Sunday.

        """
        Adding the weekly values of the summer period
        """
        data[cutoff_1:end_index_first_week_Summer] = db_week_winter_vals[start_ind_weekly_Summer:]
        data[end_index_first_week_Summer:end_index_first_week_Summer + repeat_count_Summer * len_db_weekvals] = np.tile(db_week_winter_vals, repeat_count_Summer)
        data[end_index_first_week_Summer + repeat_count_Summer * len_db_weekvals:cutoff_2] = db_week_winter_vals[:remainder_Summer] 

        """
        Adding the weekly values of the second winter period
        """
        data[cutoff_2:end_index_first_week_Winter2] = db_week_winter_vals[start_ind_weekly_Winter2:]
        data[end_index_first_week_Winter2:end_index_first_week_Winter2 + repeat_count_Winter2 * len_db_weekvals] = np.tile(db_week_winter_vals, repeat_count_Winter2)
        data[end_index_first_week_Winter2 + repeat_count_Winter2 * len_db_weekvals:] = db_week_winter_vals[:remainder_Winter2] 

        """
        Creating a dataframe from the data.
        """
        index = pd.Index(times, name=TIMESTAMP_STRING)
        db_year = pd.DataFrame(index=index, columns=[ELDEMAND_STRING], data=data)

        return db_year

    def initiate_pv_sum_win(self) -> None:
        """
        Initiates the average PV energy for the building for summer and winter.
        Technique used is ancient magic. It only works for values that exist in the database though. 
        If someone has a PV system with an angle of 32, too bad dawg. We don't do that here.
        Only steps of five for the angle and steps of 45 for the azimuth.
        Anything else is pure heresy.
        """

        if self.pv_summer is not None and self.pv_winter is not None:  # no need for all this if they already exist.
            return
        
        df_pct = get_pct_db()  # the database for the percent thingies. For each direction and angle.

        pv_systems = [pv for home in self.homes for pv in home.pv_system]  # getting the pv systems from each home in the current system of friendly neighbors.
        pv_max_powers = [pv.max_power for pv in pv_systems]  # getting the kWp for each system
        pv_max = sum(pv_max_powers)  # in kWp. Could be one step with the earlier line, but who cares?
        pv_shares = np.array(pv_max_powers)/pv_max  # adds up to 100. Needed as weight for the average.

        pv_eng_summer = np.empty(len(pv_systems), dtype=np.ndarray)  # empty for now because it's more efficient to appending two hundred thousand values (approximately).
        pv_eng_winter = np.empty(len(pv_systems), dtype=np.ndarray)  # same as above, except colder.

        for i, pv_sys in enumerate(pv_systems):  # filling our arrays with values
            pv_eng_summer[i] = df_pct.at[(SUMMER_STRING, pv_sys.azimuth, pv_sys.angle), PERCENT_STRING]  
            # searching our database for an exact match. Any innovators or creative folk that have an orientation of 72.2 can go kick rocks.
            # I could just interpolate between the values and get the exact one, but eh... Ain't got time for that, dude.
            pv_eng_winter[i] = df_pct.at[(WINTER_STRING, pv_sys.azimuth, pv_sys.angle), PERCENT_STRING]

        pv_eng_summer = pv_eng_summer*pv_shares  # weighting the values for the percentages.
        pv_eng_winter = pv_eng_winter*pv_shares  # same, but with more chilling winds.

        pv_eng_summer = sum(pv_eng_summer)  # hopefully this gives the average percent of the whole PV systems of everyone weighted by the max power.
        # Is this the best method? who knows. Not me, dude.
        pv_eng_winter = sum(pv_eng_winter)  # Same. But wind's howling.

        self.pv_summer = pv_eng_summer*pv_max  # average power in kW for each hour in the day. Only 24 values.
        self.pv_winter = pv_eng_winter*pv_max  # average power in kW for each hour in the day. Only 24 values.       

    def _get_db_week_simsmart_season(self, is_summer: bool = False):
        """
                Creates a weekly schedule of electric charging and returns the dataframe. Using 2024 because it started on a monday.
                The year doesn't get saved later and plays no part in the returned database.
        """

        """
        pv_vals dependent on if it's summer or winter.
        """
        self.initiate_pv_sum_win()
        if is_summer:
            pv_vals = self.pv_summer
        else:
            pv_vals = self.pv_winter

        """
        Initializing shit.
        """
        date_start = np.datetime64(f"{2024}-01-01T00:00")  # 2024 cause it started on Monday
        date_stop = np.datetime64(f"{2024}-01-08T00:00")  # week later unless my math is off.
        deltie = np.timedelta64(10, "m")  # 10 minutes my dawg.

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times_week = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))  # times for the week with 10 minutes as step

        demands_week = np.zeros(times_week.shape[0])  # assume no demand until proven otherwise.
        
        # indexes_car_temp = [[timey.day-1, timey.time()] for timey in times_week]  # the dataframe has the indexes day of week and time. 
        # index_week = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)

        index_week = pd.Index(times_week, name=TIMESTAMP_STRING)

        db_week = pd.DataFrame(columns=[ELDEMAND_STRING], data=demands_week, index=index_week)  # dataframe with correct index and column name but full of 0s.

        cars = self.get_all_el_cars()

        for _, car in enumerate(cars):  # since our "get_specific_weekly_stupid" method works for each car at a time, we must iterate over every car.
            updater = self._get_specific_week_simsmart(car, pv_vals)
            db_week = db_week.add(updater, fill_value= 0)  # it doesn't work right. I don't know why.

        return db_week

    @staticmethod
    def _get_specific_week_simsmart(e_car:ElCar, pv_gen: np.ndarray):
        """
        :param e_car: The electric car.
        return: Dataframe of The added charge demand caused by this car for the case of smart meter and pv and selfish user.
        Can be used for both winter or summer, since the pv generation is passed in.
        """
        rest_energy = 0
        values_car_temp = np.empty(shape=0)
        indexes_car_temp = np.empty(shape=0, dtype=np.dtypes.DateTime64DType)
        delt_10_min = np.timedelta64(10, "m")
        pv_gen = np.tile(np.repeat(pv_gen, 6), 8)  # repeat eight times in case the charging time exceeds midnight or even a few days.
        for indiana, cycle in enumerate(e_car.el_schedule):
            next_cycle = get_next_element(indiana, e_car.el_schedule)
            charge_time = get_charge_time(cycle,next_cycle)
            weekday_back, time_back = cycle.get_time_back()

            ind_time_start = int(time_back.minute/10 + time_back.hour*6)
            np_charge_time_rounded = np.timedelta64(int(round(charge_time*6, 0)*10), "m")  # We round it because the database is in intervals of 10 minutes.
            ind_time_end = int(ind_time_start + np_charge_time_rounded/delt_10_min)
            pv_gen_cycle = pv_gen[ind_time_start:ind_time_end]
            pv_gen_cycle = np.clip(pv_gen_cycle, 0, e_car.max_charge_pow)  # generation beyond the max charging capacity of the car doesn't matter.
            max_solar_expected = sum(pv_gen_cycle)/6  # divided by six because of ten minute intervals. Change kW to kWh.

            energy_need = min(cycle.distance/e_car.efficiency + rest_energy, e_car.battery_cap)  # if the car goes below zero on fuel, 
            scale_factor = min(energy_need/max_solar_expected, 1)
            pv_used_by_car = pv_gen_cycle*scale_factor

            el_car_total_demand = EnergyManagement._get_simple_smart_cycle_car_demand(pv_used_by_car, energy_need, e_car.max_charge_pow)

            rest_energy = energy_need - np.sum(el_car_total_demand)/6  # if not enough time to charge fully, the battery is left partly empty
            
            start_datetime = np.datetime64(f"2024-01-0{weekday_back+1}T{time_back}")  # starting to charge directly after getting back from the current trip.
            end_datetime = start_datetime + np_charge_time_rounded  # we assume time_back is a valid time and already rounded to the nearest 10 minutes.

            new_inds = np.arange(start=start_datetime, stop=end_datetime, step=delt_10_min)  # these should be all the timestapmps where charging occurs.
            indexes_car_temp = np.append(indexes_car_temp, new_inds)  # adding our new timestamps into the indexes array.
            values_car_temp =  np.append(values_car_temp, el_car_total_demand)  # the demand is constant and therefore can be applied to every timestamp in the indexes.

        
        """
        Pushing the dates that wrap around the week back to their rightful place.
        """
        threshold_date = pd.to_datetime('2024-01-08')
        indexes_car_temp = np.array(indexes_car_temp, dtype=np.dtypes.DateTime64DType)
        start_index = np.searchsorted(indexes_car_temp, threshold_date)  # I*ll be honest with you, I don't remember how this works.
        if start_index < len(indexes_car_temp)-1:  # no clue.
            week_offset = np.timedelta64(7, "D").astype('datetime64[D]')
            indexes_car_temp[start_index:] = indexes_car_temp[start_index:] - week_offset  # It tries to push back anything past- 
            # a week to the start of the month, but does it work? no idea.

        """
        creating the dataframe.
        """
        # indexes_car_temp = [[d.day-1, d.time()] for d in indexes_car_temp]  # the dataframe has the indexes day of week and time. 
        # # 2024 starts on monday, so we push back the day of month by one and get the day of the week.

        # index = pd.MultiIndex.from_tuples(indexes_car_temp, names=WEEK_INDNAMES)
        
        index = pd.Index(indexes_car_temp, name=TIMESTAMP_STRING)

        db_week_car_update = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=values_car_temp)  # we use update because- 
        # otherwise the database should be full of zeros when the car isn't charging.
        return db_week_car_update

    @staticmethod
    def _get_simple_smart_cycle_car_demand(pv_used_by_car: np.ndarray, energy_need: float, max_charge: float):
        """
        iterates until satisfaction and returns the demand array for the car cycle. We assume the car can be physically fully charged within the timeframe.
        pv_used_by_car: the array for the pv generation expected to be used by the car. It should have the same length as the expected values.
        energy_need: the amount of energy the car needs in this cycle in kWh.
        max_charge: the max charging power for the car in kW.
        """      
        el_car_demand = pv_used_by_car.copy()
        while True:
            remaining_demand = energy_need - sum(el_car_demand)/6  # divide by six for kW to kWh conversion
            if round(remaining_demand, 2) == 0:
                return el_car_demand
            
            el_car_demand += remaining_demand*6/len(el_car_demand)
            el_car_demand = np.clip(el_car_demand, 0, max_charge)

    def _get_yearly_genius_no_battery(self, year):
        """
            The magnum opus of this entire dumbass project. Probably won't be ready in time, because I ain't got no time.
        """

        """
        Initializing usual stuff.
        """
        battery_capacity = self.get_battery_cap()
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))

        cars: list[ElCar] = self.get_all_el_cars().tolist()
        data = [car.df for car in cars]
        current_timestamp = date_start
        df_cars = pd.DataFrame(index=times, data=0)
        energy_needs = np.array(np.tile(None, len(cars)), dtype=float)
        times_left = np.array(np.tile(None, len(cars)), dtype=float)
        df_sol = get_solar_db()
        pv_gen = self.get_pv_generation_year()
        db_year = 0
        """
        First week
        """
        for i in range(len(pv_gen)):
            """
            Getting ready
            """
            pv_gen_now = pv_gen[i]  # pv generation at this very moment.
            pv_available_for_car = pv_gen_now  # initializing the pv_available_for_car that will later be substracted every time a car uses some of the energy.
            cars_in_garage = np.array([car.is_in_garage(current_timestamp) for car in cars])  # getting back a mask to know which cars are in the garage and which are away.
            """
            Noning those who must be noned. Cars that are not in the garage are not cars we can charge and so we don't look at them.
            """
            energy_needs[not cars_in_garage] = None
            times_left[not cars_in_garage] = None
            """
            If all cars are away we can't do anything anyway except load the battery and this is the scenario without battery.
            """
            if len(cars_in_garage) == 0:
                continue
            """
            Filtering the cars and getting back a priority list based on who leaves the garage next. Whoever leaves earliest gets priority. 
            If there's a tie, there is a shirtless chess match do decide the winner.
            """
            filtered_cars: list[ElCar] = np.array(cars)[cars_in_garage].tolist()
            priority = np.argsort([car._get_charge_time_left(current_timestamp) for car in filtered_cars])
            filtered_needs = energy_needs[cars_in_garage]  # how much energy the car still needs until it's fully charged, in kWh.
            """
            Iterating over the cars to divide the pv_generation for.
            """
            for ind in priority:
                if pv_available_for_car == 0:  # no point in looking at the cars and analyzing all this stuff if there is no generation left anyway. 
                    # Sure, there might be a battery, but this is the method without a battery.
                    break  
                car = filtered_cars[priority]
                energy_need = energy_needs[priority]
                max_charge = min(car.max_charge_pow, energy_need/times_left[priority])
                pv_used_by_car = min(pv_available_for_car, car.max_charge_pow)
                  
            for i, time_left in enumerate(times_left):
                if time_left is None:  # if time_left or filtered_needs are none, it means they need to be initialized because the car just arrived at the garage.
                    filtered_needs[i] = filtered_cars[i]._get_energy_needed(current_timestamp)
                    times_left[i] = filtered_cars[i]._get_charge_time_left(current_timestamp)
                else:
                    times_left[i] = times_left - 1/6  # adjusting for the fact that ten minutes have passed.


            current_timestamp += deltie
            energy_needs[cars_in_garage] = filtered_needs
            return


        return db_year
    
    def get_all_el_cars(self) -> np.ndarray[ElCar]:
        """
        Returns an array of all the cars.
        """
        cars = np.array([car for home in self.homes for car in home.e_cars])
        return cars
    
    def get_all_el_cars_in_garage(cars: np.ndarray[ElCar], cur_timestamp):
        return

    def _get_first_week_genius(self):
        return
    
    def _get_yearly_genius_with_battery(self, year):
        """
            A function for the summer demand in case of smart meters in the simple method of prioritizing times between
            07:00 and 16:00. In this method the charging schedule is exact for every week.

            :param year: the year you want printed.
            :return: The summer charging schedule of the electric car, prioritizing times with solar radiation.
        """

        """
        Initializing usual stuff.
        """
        bat_cap = self.get_battery_cap()
        self.initiate_pv_sum_win()
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")
        current_date = date_start
        """
        First week
        """
        index, values = self._get_first_week_genius
        while True:
            return


        return db_year

    def get_yearly_el_car_schedule(self, assume_full: bool = True, year: int = DEFAULT_YEAR):
        """
        returns the yearly schedule for all electric cars in this system.
        """
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))
        cars: list[ElCar] = self.get_all_el_cars().tolist()
        weekly_demands = np.array([car.df_schedule.loc[:, EL_CAR_ENERGY_NEEDED_STRING].to_numpy() for car in cars])
        yearly_demands = np.array([self.get_yearly_base_demand(weekly_demand, year) for weekly_demand in weekly_demands])
        db_year = pd.DataFrame(index=times, data=yearly_demands, columns=[i for i in range(len(cars))])

        if assume_full:
            first_index_zero = [get_first_index(weekly_demand, 0) for weekly_demand in weekly_demands]
            for i in range(first_index_zero):
                end_index = first_index_zero[i][0]
                yearly_demands[: first_index_zero] = 0

        return db_year
    
    @staticmethod
    def get_yearly_demand_from_weekly_demand(weekly_vals: np.ndarray, year: int = DEFAULT_YEAR) -> np.ndarray:
        """
        changes a weekly schedule into a yearly schedule by repeating it a bunch of times.
        """
        date_start = np.datetime64(f"{year}-01-01T00:00")
        date_stop = np.datetime64(f"{year+1}-01-01T00:00")
        weekday_start = pd.to_datetime(date_start).weekday()
        deltie = np.timedelta64(10, "m")

        # times = pd.timedelta_range(date_start, date_stop, freq=deltie).to_series()
        times = pd.DatetimeIndex(np.arange(start=date_start, stop=date_stop, step=deltie, dtype=np.datetime64))

        db_week_vals = weekly_vals
        len_db_weekvals = len(db_week_vals)
        datalength = len(times)
        data = np.empty(dtype=float, shape=datalength)
        
        start_ind_weekly = int(len_db_weekvals/7*weekday_start)  # If we start on monday, this is zero. If we start on tuesday this is 24*6=144 if the intervals are ten minutes. 
        end_index_first_week = len_db_weekvals - start_ind_weekly  # This is the index for the end of week 1 of the year. It should be the length of a week minus the steps we skip.
        rest_datalength = datalength - end_index_first_week

        repeat_count = rest_datalength // len_db_weekvals 
        remainder = rest_datalength % len_db_weekvals

        data[:end_index_first_week] = db_week_vals[start_ind_weekly:]
        data[end_index_first_week:end_index_first_week + repeat_count * len_db_weekvals] = np.tile(db_week_vals, repeat_count)
        data[end_index_first_week + repeat_count * len_db_weekvals:] = db_week_vals[:remainder] 

        index = pd.Index(times, name=TIMESTAMP_STRING)
        db_year = pd.DataFrame(index=index, columns=[ELDEMAND_STRING], data=data)

        return db_year


def get_amortization_time_pv(pv_cost: int, yearly_el_self_use: float|int, yearly_el_feed_in: float|int, price_electricity: float|int, pv_max_power, opp_cost: float = 0.05):
    """
    PV cost in euros. Self use in kWh. Feed in in kWh. Price in Cent/kWh. Max power in KWp. Opportunity cost in percent.
    """
    """
    Getting the correct tariff.
    """
    if yearly_el_self_use == 0:
        feed_in_tariff = get_pv_feed_in_tariff(pv_max_power, full_feed_in=True)
    else:
        feed_in_tariff = get_pv_feed_in_tariff(pv_max_power, full_feed_in=False)
    """
    Unit conversions.
    """
    yearly_el_self_use = yearly_el_self_use/1000  # from kWh to MWh
    yearly_el_feed_in = yearly_el_feed_in/1000  # from kWh to MWh
    price_electricity = price_electricity*10  # from cent/kWh to Euro/MWh

    el_savings = yearly_el_feed_in*price_electricity  # Euro/MWh * MWh = Euro
    el_earnings = yearly_el_feed_in*feed_in_tariff  # # Euro/MWh * MWh = Euro
    pv_total_yearly_benefit = el_savings + el_earnings

    if pv_total_yearly_benefit <= opp_cost * pv_cost:
        raise ValueError("The investment will never fully amortize.")

    years = -np.log(1 - (opp_cost * pv_cost) / pv_total_yearly_benefit) / np.log(1 + opp_cost)
    return years



def get_pv_feed_in_tariff(max_power: float|int, full_feed_in: bool = False):
    """
    Gives it back in Euro per MWh
    """
    if full_feed_in:
        pv_thresh = PV_THRESHOLDS_FULL_FEED_IN
        pv_tariffs = PV_TARIFFS_FULL_FEED_IN
    else:
        pv_thresh = PV_THRESHOLDS_PART_FEED_IN
        pv_tariffs = PV_TARIFFS_PART_FEED_IN
    
    if max_power < pv_thresh[0]:
        return pv_tariffs[0]
    if max_power > pv_thresh[-1]:
        return 0
    temp_tariff = pv_tariffs[0]*pv_thresh[0]/max_power
    for i in range(1, len(pv_tariffs)):
        if max_power < pv_thresh[i]:
            temp_tariff += pv_tariffs[i]*(max_power - pv_thresh[i-1])/max_power
            return temp_tariff
        temp_tariff += pv_tariffs[i]*(pv_thresh[i] - pv_thresh[i-1])/max_power
        
    return


def get_num_modules(power_needed: float, power_per_module: float):
    """
    power needed in kW
    power per module in kW
    returns the numner of modules needed to fulfill the law. Provided the nominal power of a module.
    """
    power_needed = round(power_needed, 1)  # no need for excessive precision.
    if power_needed % power_per_module == 0:  # if you can divide exactly.
        num_modules = int(power_needed/power_per_module)
    else:
        num_modules = int(power_needed//power_per_module + 1)
    pv_power = num_modules*power_per_module

    print(f"you need {num_modules} modules, which is the equivalent of {pv_power} kWp.")
    return num_modules

# heaty = HeatingSystem(0, 0, 2, 1998, 32, 10000)
# logging.debug(heaty.help_electricity)
# heaty2 = HeatingSystem(2, 5, 0, 2023, 32, 40000)
# homey = Home(460, 800, 0.5, 1975, heaty)
# print(homey)
# homey.set_new_heating_system(heaty2)
# print(homey)
# print(homey.get_subsidy())


    # def get_stupid_smart_demand(self, datetime_start: np.datetime64, datetime_end: np.datetime64):
    #     """
    #     Gives back the energy demand in the next whatever amount of time between datetime_end and datetime_end.
    #     """
    #     max_demando = [house.get_yearly_base_el_demand() for house in self.homes]  # the max demands for each house
    #     max_demando = sum(max_demando)
    #     dif = datetime_end - datetime_start  # the timedelta between start and end.
    #     hours = dif/np.timedelta64(1, "h")

    #     return max_demando/DEMAND_HOURS*hours  # we divide by DEMAND_HOURS because we assume a higher demand for when the electric car is charging. 
    #     # Since it is usually during the day when the PV is providing electricity. It is 20/06/24 4000 hours. So about double the average.
    
    # def get_simple_smart_chargeplan(self, time_now: datetime.datetime, pv_vals:np.ndarray):
    #     max_solar = self._get_simple_smart_maxsolar(np.datetime64(time_now), np.datetime64(self.next_time_back), pv_vals)
    #     demand = self.get_stupid_smart_demand(time_now, self.next_time_back)
    #     max_charge = 0
    #     min_charge = self.get_min_charge()
    #     charge_plan = pd.DataFrame()
    #     return charge_plan

    # def _get_simple_smart_maxsolar(self, datetime_start: np.datetime64, datetime_end: np.datetime64, pv_vals:np.ndarray):
    #     max_solar = 0
    #     indies = pd.date_range(start=datetime_start, end=datetime_end, freq="10T").to_series()

    #     for ind, val in indies.items():
    #         max_solar += pv_vals[val.hour]/6

    #     return max_solar

    # def _simple_smart_departure(self, e_car: ElCar) -> None:
    #     """
    #     smart_smart car goes on a trip. So we update the status of it being home. The next driving cycle is becoming the
    #     current. And the next is now the one after that.
    #     """
    #     e_car.demand = 0
    #     e_car.car_home = False
    #     e_car.cur_cycle = e_car.next_cycle
    #     e_car.next_cycle = get_next_element(e_car.cyc_ind, e_car.el_schedule)
    #     e_car.cyc_ind += 1
    #     e_car.charge_plan = None

    # def _simple_smart_got_back(self, e_car: ElCar, time_now: datetime.datetime, pv_vals:np.ndarray) -> None:
    #     """
    #     smart_smart car gets back from trip. So we update the battery level by the energy the trip needed.
    #     """
    #     e_car.battery_level = max(e_car.battery_level - e_car.cur_cycle.distance / e_car.efficiency, 0)
    #     e_car.next_time_back, e_car.next_time_departure = get_times_next_cycle_weekly(e_car.next_cycle)
    #     e_car.car_home = True
    #     e_car.charge_plan = self.get_simple_smart_chargeplan(e_car.next_time_departure, pv_vals)
    #     e_car.demand = self._get_current_simple_smart_demand(e_car.battery_level, time_now, pv_vals)

    # def _simple_smart_charge(self, e_car:ElCar) -> None:
    #     """
    #     smart_smart car is home and the battery is not full yet, so we charge and apply some logic on the bitch.
    #     """
    #     e_car.battery_level += e_car.demand / 6  # demand is in kW and battery in kWh.
    #     # Since the time steps are ten minutes, we divide by 6.
    #     if round(e_car.battery_level, 2) == e_car.battery_cap:  # if the battery is now full, the demand is 0.
    #         # We round it in case of rounding errors.
    #         e_car.battery_level = e_car.battery_cap
    #         e_car.demand = 0
    #     else:  # if the battery is not full, we find the next demand for the next ten minutes.
    #         pass

    # def _get_current_simple_smart_demand(self, time_now: datetime.datetime,
    #                                      pv_vals:np.ndarray):
    #     """
    #     :return: The current demand for a non-smart meter case.
    #     """
    #     if time_now.time() < datetime.time():
    #         pass
    #     energy_need = self.battery_cap - self.battery_level
    #     demand = min(self.max_charge_pow, energy_need * 6)  # times six to get the kW needed to cover the
    #     # energy needed in ten minutes. In reality the car would charge with max power for two minutes for
    #     # example, but since we work with ten minute values, we spread the energy demanded over the full ten
    #     # minutes.
    #     return demand
# self.el_schedule = sorted(self.el_schedule, key=lambda cyc: (cyc.weekday, cyc.departure))
        # self.cur_cycle = self.el_schedule[-1]
        # self.next_cycle = self.el_schedule[0]
        # self.cyc_ind = 0
        # self.battery_level = self.get_battery_cap()

        # self.next_time_back, self.next_time_departure = get_times_next_cycle_weekly(self.next_cycle)

        # datey_time = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
        # deltie = datetime.timedelta(minutes=10)
        # data = []
        # indies = []
        # self.demand = 0
        # self.car_home = True
        # self.charge_plan = None

        # """
        # Going through the week in ten minute steps, starting from midnight going into Monday.
        # """

        # while datey_time.day < 8:
        #     cur_sol = pv_vals[datey_time.time().hour]
        #     indies.append((datey_time.weekday(), datey_time.time()))

        #     if datey_time == self.next_time_departure:
        #         self._simple_smart_departure()
        #     if datey_time == self.next_time_back:
        #         self._simple_smart_got_back(datey_time, winter_vals)
        #     if self.car_home and self.battery_level != self.battery_cap:
        #         self._simple_smart_charge()

        #     data.append(self.demand)
        #     datey_time += deltie

        # index = pd.MultiIndex.from_tuples(indies, names=WEEK_INDNAMES)
        # db_week = pd.DataFrame(columns=[ELDEMAND_STRING], index=index, data=data)
        # return db_week
