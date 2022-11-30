import pandas as pd
from matplotlib import pyplot as plt


def numerate_mission_status(df):
    """
    Takes the "MissionStatus" Column of the Space Data, and converts the rows containing "Success" to 1 and the rows containing the words
    "Failure", "Prelaunch Failure", and "Partial Failure" to 0.

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    :return: The same dataframe with the modified "MissionStatus" column.
    :rtype: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    for item in df.index:
        if df['MissionStatus'][item] == "Success":
            df['MissionStatus'][item] = 1
        else:
            df['MissionStatus'][item] = 0
    return df


def plot_top_5_most_used_LVs(df):
    """
    Plots the bar-chart showing the top 5 most heavily used Launch Vehicles and the total number of missions in which they have been used.

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    Launch_vehicle_counts = df['LaunchVehicle'].value_counts()

    top_5_LVs = list(Launch_vehicle_counts.index)[:5]
    top_5_LV_missions =  Launch_vehicle_counts.to_list()[:5]

    fig, axs = plt.subplots(1, figsize = (10,7))
    axs.bar(x = top_5_LVs, height = top_5_LV_missions, color = "green")
    axs.set_title('Missions of Top-5 Launch Vehicles')


def plot_success_rate_LVs(df):
    """
    Plots the horizontal bar-graph showing the success-rate of Launch Vehicles which have been used in 30 missions or more.

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    success_rate_launch_vehicles = []

    Launch_vehicle_counts = df['LaunchVehicle'].value_counts()

    for item in Launch_vehicle_counts.index:
        if Launch_vehicle_counts[item] >= 30:
            success_rate_item = sum(df.loc[(df['LaunchVehicle'] == item), 'MissionStatus'].tolist()) / \
                                Launch_vehicle_counts[item]
            success_rate_launch_vehicles.append((item, success_rate_item))

    success_rate_launch_vehicles = sorted(success_rate_launch_vehicles, key=lambda x: x[1], reverse=True)

    y1, width1 = [], []

    for key, value in success_rate_launch_vehicles:
        y1.append(key)
        width1.append(value)

    fig, axs = plt.subplots(figsize=(10, 10))
    axs.barh(y=y1, width=width1, color="green")
    axs.set_title('Success rate of Launch Vehicles')


def most_widely_used_LVs(df):
    """
    We have found in our dataset that the maximum number of different organizations a single Launch Vehicle model has been used in is 3. This
    function displays the list of those Launch Vehicles which has been used in 3 different organizations.

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    :return: The list of those Launch Vehicles which has been used in 3 different organizations.
    :rtype: list
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    Unique_companies = []

    Launch_vehicle_names = df.LaunchVehicle.unique()

    for i in Launch_vehicle_names:
      Company_counts = len(df.loc[(df['LaunchVehicle']==i),'Company'].unique())
      Unique_companies.append((i,Company_counts))

    Unique_companies = sorted(Unique_companies, key = lambda x: x[1], reverse = True)
    launch_vehicles_most_used = []
    for i in Unique_companies:
      if i[1] > 2:
        launch_vehicles_most_used.append(i[0])

    return launch_vehicles_most_used


def plot_LVs_per_country(df):
    """
    Plots a horizontal bar-graph showing the total number of Launch Vehicles used by a country (or a company based in it) over the years.

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    Unique_launch_vehicles_per_country = []

    Country_names = df.Country.unique()

    for i in Country_names:
      if i is not None:
        i_count = len(df.loc[(df['Country']== i),'LaunchVehicle'].unique())
        Unique_launch_vehicles_per_country.append((i,i_count))


    Unique_launch_vehicles_per_country = sorted(Unique_launch_vehicles_per_country, key = lambda x:x[1], reverse = True)

    y1, width1 = [], []
    for i in Unique_launch_vehicles_per_country:
      if (i[0] == "Kiritimati"):
        continue
      y1.append(i[0])
      width1.append(i[1])

    fig, axs = plt.subplots(figsize = (10,7))
    axs.barh(y = y1, width = width1, color = "green")
    axs.set_title('Number of Different Launch Vehicles Used Per Country')


def plot_Missions_per_country(df):
    """
    Plots a horizontal bar-graph showing the number of space missions conducted within a country
    (either by federal organizations or the companies based within it).

    :param df: The input dataframe containing the space data
    :type df: pd.DataFrame
    """
    assert isinstance(df, pd.DataFrame) and len(list(df.index)) != 0

    temporary_series = df['Country'].value_counts()
    Countries, Missions = [], []
    for i in temporary_series.index:
        Countries.append(i)
        Missions.append(temporary_series[i])

    fig, axs = plt.subplots(figsize=(10, 7))

    axs.barh(y=Countries, width=Missions, color="green")
    axs.set_title('Number of Missions Per Country')