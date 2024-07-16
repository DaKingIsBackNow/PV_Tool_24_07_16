import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DB = pd.read_csv("H:/Dokumentos/Alt/stundenwerte_TU_03987_hist/produkt_tu_stunde_18930101_20211231_03987.txt"
                 , sep=";")
DB.columns = ["S", "D", "Q", "T", "H", "end"]  # S=Station, D=Date, Q=Quality, T = Temperature H=Humidity
DB["D"] = DB["D"].apply(str)
DB["T"] = DB["T"].apply(str)
DB["Year"] = DB["D"].str[:4]
DB["Month"] = DB["D"].str[4:6]
DB["Day"] = DB["D"].str[6:8]
DB["Hour"] = DB["D"].str[8:10]
DB = DB[["Year", "Month", "Day", "Hour", "T"]]
DB["Year"] = DB["Year"].apply(int)
# DB = DB[DB["Year"] >= 1980]
# DB.index = range(len(DB.index))

# DB.to_csv("H:/Dokumentos/Alt/Temperature.csv", index=False, sep=";")
# print(DB.head())

DB["T"] = DB["T"].apply(lambda x: x.replace(',', '.'))
DB["T"] = DB["T"].apply(float)
y = []
x = np.arange(1893, 2022, 1)
for year in x:
    y.append(DB.loc[DB['Year'] == year]["T"].mean())

plt.plot(x, y, 'r')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# add trendline to plot
plt.plot(x, p(x))
for i, val in enumerate(x):
    print(val, y[i])
plt.show()

# DB = DB.sort_values(by='T', ascending=True)
# print(DB.head(20))

