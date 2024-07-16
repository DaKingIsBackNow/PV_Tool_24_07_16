# PV_Tool_24_07_16
The PV tool. It does PV things.

# Key features
* Gives back amount of self consumption from pv, amount fed to the grid, and amount consumed from the grid. 
* Based on a bunch of variables like the electric cars, pv systems, smart meters and so on.
* Can do some financial calculations once you know where to look.
* Finds the legally necessary amount of PV needed for a residential building, based on its attributes.

# Installation
run the requirements.txt thingy in the main directory. pip install -r requirements.txt or something like that.

# Starting out with the main directory:
* Everything outside of the directory main_pv is kind of irrelevant. Except for "main.py". You kind of need that. "old_main.py isn't needed though. You can delete that.
* A bunch of testing python scripts. Feel free to ignore those.

# CSV files in main directory:
* simulation_results, simulation_results_monthly, and simulation_results_monthly_old are results of two example buildings with certain assumptions and all that.
* simulation_results_old is calculated wrong. Feel free to delete it. I'm just a hoarder.

# "main_pv" folder:
Go inside the main_pv folder to see the important parts. There you can see: 
## "classes.py"
This holds all the classes I have created and some other functions for PV systems.
## "fix_dbs.py":
This was to take unprocessed databases from the DWD and make them into csv files I can work with. There's a lot of stuff commented out that I used before and then didn't need anymore.
I'm pretty sure you don't need any functions above "get_len_float_int_array". The only one that matters is "solar_year" that then calls other functions it needs to add the outside temperature and wind speed and all that.
print(7) is also crucial, obviously. Rest of the commented out stuff is me being scared of needing it later.

## "simulate.py"
I'm going to be honest. I forgot this even existed. You can just delete it.

## "solar_functions.py"
This is where we do a lot of solar calculations with the help of pvlib. It was supposed to be functions independent of pvlib, but results made no sense, so here we are.

## "solar_validations.py"
This is like the name suggests, a bunch of validations for the solar functions. Doesn't really do anything since we just use pvlib now. But hey, it shows effort, right? besides the error comments are amusing.

## "standard_variables.py"
This one is for most of the assumptions and strings we need many times. I probably forgot some assumptions or naked strings laying in the code, but I tried to consolidate them here.

## "test.py"
Just an old testing place. Feel free to abolish it.
## "tester.py"
Another old testing place. Can be destroyed too.

## "databases" directory:
Another mess, eh? Don't worry. Detective Danny is here to walk you through it and be equally as confused.

### The three main ones
There are three main databases from the DWD for the solar data, temperature and wind speed. You can figure out which one is which by their names. These are unprocessed and become a good little csv file after being processed by fix_dbs.py.

### "helper" directory
Has one csv file called test_year_stupid.csv. As you can assume, this isn't relevant and can be deleted.

### "main" directory
This is where the real databases hang out. 
* "Solar4928-2023.csv" This is the main database for the solar radiation, wind speed and outside air temperature. 4928 is the number of station it's taken out of and 2023 is the year obviously.
* "ElDemand.csv" is the electric demand over the year for a consumption of 1000 kWh. We multiply it by a factor to get the base demand for a specific building. 
Does not include any heating systems like heat pumps, nor does it include consumption of electric cars. Those profiles are generated independently.
* "EngSrcTypes.csv" are the energy sources. Electricity, oil, sun, and all that. You have eyes, you can look at it.
* "MainTypes.csv" are the main types of heating systems.
* "SubTypeNames.csv" is the connection between the main heating types and their subtypes. A heating system is always a subtype of a main type. Like a condensing boiler which is a type of boiler.
* "SubTypes.csv" is the connection between the sub types and energy sources. It also has the default thermal efficiencies in there. Help energy in percent of max power. 
For example a help energy of 0.01 for a 100 kW boiler would mean a constant demand of 1 kW for the circulation pumps and regulation. Is that realistic? hell no. But what do you want me to do?
* "SubsidyDesc.csv" Here you can find the index of the subsidies, the base percentage of them, the extra percentage you get for replacing oil/coal old gas boilers, and the descrpition.
The new GEG2024 completely changed the subsidies though, so this is kind of worthless.
* "Temperature4928-2023.csv" is old. Feel free to delete it. The temperature is in the solar database anyway.

### "seasons" directory
* "PCTs.csv" PCTs is short for percentages. This is where the seasonal average pv generation as percent of maximal nominal generation is stored. For every full hour of the day there is a corresponding value. 
The rows have varying compass directions and module angles. This is used for the stupider method for planning a charging schedule for electric cars with pv and a smart meter system.
You still need this database though, because the smarter algorithm isn't ready yet.
* "Summer.csv" and "Winter.csv" are old. You can delete them. The summer and winter are both in PCTs because why two databases?

# "WP" directory
This is also very old. It isn't integrated in the calculations. These three modules try to model a specific heat pump based on an external excel table for the COP values.
This could be used in the future for more accurate predictions for heat pumps with data sheets. But I would probably start over and avoid confusion.

# main.py
This is where you're supposed to create your buildings and assumptions and simulate them. This tool simulates two buildings and their combined system. 
Changing the amount of buildings or systems shouldn't be hard but saving the results to a database can't be achieved with the current functions for other possibilities other than two buildings and a system.
You can change the assumptions for the first two buildings to get a feel for it first.

* Start at the top right below the last function and work your way down.
* Building variables: You can see a few tuples for example for the smart meters of the two buildings. You have comments to guide you in the code. Enter what you want the buildings to have in the calculation.
* Only change the values in the same format they are given as. You can't add more pv systems for example in this way. But you can add more manually later down the code.
* After you input the data needed for the pv systems, you can see the initiation of the variables and all that. If you feel like you've figured it out, feel free to change stuff there too. Like the schedules of the cars, or more pv systems.
Many variables don't matter and change nothing. Like the building year for pv systems or heating systems. The return temperature changes nothing. The costs of the heating systems change nothing and so on.
* At the end of the initialization, you can find results section. Currently the code will save all variations of the current two buildings for yearly values and monthly values. It also calculates the amount of pv_needed for the buildings.
Feel free to comment out stuff you don't want to currently do. If you just want to run one scenario and get the results, you don't need to call the yearly and monthly scnearios functions.

# Disclaimer
* Comments are not professional and include some profanity. This was a personal project that wasn't supposed to ever be public.
* Also, my spelling might suck. Try to ignore that.
* Don't expect the fix_csv to work with different looking databases or any of that. This code was made quickly to work, not to be flexible.
* You may wonder why there are so many things I tell you to delete but don't do so myself. The answer is that I don't want to accidentally break the program. I don't think I need those things. But what if I secretly do?
# Licence 
Do whatever you want with it. Props to you if you manage to get any meaningful use out of this mess. No licence here.
