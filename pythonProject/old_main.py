# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Strg+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


def get_winter_avg():
    path = "Databases/Seasons/Winter.csv"
    df = pd.read_csv(filepath_or_buffer=path)
    avg = df["Solarstrahl"].values
    return avg


print(get_winter_avg())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
