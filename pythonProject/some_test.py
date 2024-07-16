
import pandas as pd
import datetime
# df = pd.read_csv(r'h:\\Studienarbeit\\PycharmProjects\\pythonProject\\Homes\\Databases\\Helper\\test_year_stupid.csv', delimiter=";", index_col = (0, 1, 2))

# df.index.names = ["Date", "Time"]
# df_reset = df.reset_index()  # Temporarily reset the index

# Now you can access the levels as columns:
# df_reset['Time'] = pd.to_datetime(df_reset['Time'])
# df_reset['Date'] = pd.to_datetime(df_reset['Date'])
# df_reset['Hour'] = df_reset['Time'].dt.hour
# df_reset = df_reset[df_reset['Hour'] == 10]

# # You can set the MultiIndex back if you need it later:
# df = df_reset.set_index(['Date', 'Time'])  


# df = df.xs(4, level="DayOfWeek")
# time_values = df.index.get_level_values("Time").values
# df_filtered = df[(pd.to_datetime(time_values).dt.hour == 10)]

# df["Time"] = pd.to_datetime(df["Time"])
# df["Date"] = pd.to_datetime(df["Date"])
# df["Hour"] = df.loc["Time"].dt.hour
# df = df.loc[df["Hour"] == 10]

# df = df.loc[df["Date"] == "2024-01-06"]
# df = df.loc[df["Date"] == 2]

# print(df.head())

print(type((10, 15, "were")))
