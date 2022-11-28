import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_dataframe(path):
    '''
    Returns a dataframe read from the given path
    @param path: path of the csv file
    @type path: str
    '''
    assert isinstance(path, str)
    df = pd.read_csv(path)
    return df


def drop_columns(df, columns):
    '''
    Returns the dataframe after removing the given columns

    @param df: Dataframe
    @param columns: list of column names to be dropped
    @type df: pd.DataFrame
    @type columns: List

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, list)

    return df.drop(columns, axis=1)


def rename_columns(df, columns):
    '''
    Returns the dataframe after renaming the given columns

    @param df: Dataframe
    @param columns: dictionary with keys as old column names and values as the new column names
    @type df: pd.DataFrame
    @type columns: Dict

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(columns, dict)

    return df.rename(columns, axis=1)


def convert_str_float(df, col_name):
    '''
    Returns the dataframe with the input column type-casted to float
    Note: Since the data had the char ',' in the numbers, replace function is used to remove the special character

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @type df: pd.DataFrame
    @type col_name: string
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)

    df[col_name] = df[col_name].apply(lambda x: str(x).replace(',', ''))
    df[col_name] = df[col_name].astype(np.float64)
    return df[col_name]


def split_date(df, col_name, num=2):
    '''
    Returns the dataframe with the date column split into year, month and date column based on num

    Note: if num=1, then only year col is added. If num=2, year and month columns are added. If num=3, year,month and day columns are added
          Column names = 'Year', 'Month' (in 3-char format) and 'Day'

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @type df: pd.DataFrame
    @type col_name: string

    '''

    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)
    assert num in [1, 2, 3]

    df[col_name] = pd.to_datetime(df[col_name], infer_datetime_format=True)
    if num == 1:
        df["Year"] = df[col_name].apply(lambda x: x.year)
    elif num == 2:
        df["Year"] = df[col_name].apply(lambda x: x.year)
        df["Month"] = df[col_name].apply(lambda x: x.strftime("%b"))
    elif num == 3:
        df["Year"] = df[col_name].apply(lambda x: x.year)
        df["Month"] = df[col_name].apply(lambda x: x.strftime("%b"))
        df["Day"] = df[col_name].apply(lambda x: x.strftime("%d"))
    df = df.drop([col_name], axis=1)
    return df


def location_split(x, delim):
    '''
    Returns a list with elements delimited by the given delim

    @param x: Input string
    @param delim: delimiter
    @type x: string
    @type delim: string

    '''

    assert isinstance(x, str)
    assert isinstance(delim, str)

    t = x.split(delim)
    if (len(t) == 3):
        t.insert(2, "")
    elif (len(t) == 2):
        t.insert(0, "")
        t.insert(2, "")
    elif (len(t) == 1):
        t.insert(0, "")
        t.insert(1, "")
        t.insert(2, "")
    t = [x.strip() for x in t]
    return t


def fill_empty_with_NaN(df, col_name, old_value):
    '''
    Returns the dataframe with the date column split into year, month and date column based on num

    Note: if num=1, then only year col is added. If num=2, year and month columns are added. If num=3, year,month and day columns are added
          Column names = 'Year', 'Month' (in 3-char format) and 'Day'

    @param df: Input dataframe
    @param col_name: Column name to be type-casted to float
    @param old_value: value to be replaced
    @type df: pd.DataFrame
    @type col_name: string
    @type old_value: string
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(col_name, str)
    assert isinstance(old_value, str)

    df[col_name] = df[col_name].replace(old_value, np.nan)
    return df[col_name]


def pre_processing(path):
      '''
      Returns the final dataframe after performing all data cleaning and pre-procesing steps
      @param path: path of the csv file
      @type path: str

      '''
      #loading the dataframe from csv file
      space_data = load_dataframe(path)

      #Remove irrelevant columns
      space_data = drop_columns(space_data,['Unnamed: 0','Unnamed: 0.1'])

      #Renaming column names
      space_data=rename_columns(space_data,{'Company Name':'Company','Status Rocket':'RocketStatus',' Rocket':'MissionCost','Status Mission':'MissionStatus'})

      #cast MissionCost column to float type
      space_data['MissionCost'] = convert_str_float(space_data,'MissionCost')

      #Splitting the Detail column into two: Launch vehicle name and Rocket name
      space_data[['LaunchVehicle', 'RocketName']] = space_data['Detail'].str.split('|',expand=True)
      space_data=space_data.drop(['Detail'],axis=1)

      #Splitting the Datum column to month and year
      space_data = split_date(space_data,'Datum',2)

      #Function to split the Location column
      space_data["Location"] =  pd.Series([location_split(x,',') for x in space_data['Location']])
      split_df = pd.DataFrame(space_data['Location'].to_list(),columns=("LaunchCenter","SpaceCenter","State/Region","Country"))
      space_data = pd.concat([space_data, split_df], axis=1)
      space_data = space_data.drop(['Location'],axis=1)

      #merge Partial and Prelaunch failure categories
      space_data['MissionStatus'] = space_data['MissionStatus'].replace({'Prelaunch Failure':'Failure','Partial Failure':'Failure'})

      #custom mappings!
      #map the below names to repsective countries

      #New mexico
      space_data.loc[(space_data["Country"] == "New Mexico"),"State/Region"] = "New Mexico"
      space_data.loc[(space_data["State/Region"] == "New Mexico"),"Country"] = "USA"

      #Launch Plateform, Shahrud Missile Test Site
      space_data.loc[(space_data["Country"] == "Shahrud Missile Test Site"),"SpaceCenter"] = "Shahrud Missile Test Site"
      space_data.loc[(space_data["SpaceCenter"] == 'Shahrud Missile Test Site'),"Country"] = "Iran"
      space_data.loc[(space_data["SpaceCenter"] == 'Shahrud Missile Test Site'),"LaunchCenter"] = "Launch Plateform"

      #Tai Rui Barge, Yellow Sea, China
      space_data.loc[(space_data["Country"] == 'Yellow Sea'),"State/Region"] = "Yellow Sea"
      space_data.loc[(space_data["State/Region"] == 'Yellow Sea'),"Country"] = "China"
      space_data.loc[(space_data["State/Region"] == 'Yellow Sea'),"LaunchCenter"] = "Tai Rui Barge"
      space_data.loc[(space_data["State/Region"] == 'Yellow Sea'),"SpaceCenter"] = ""

      #LP-41, Kauai, Pacific Missile Range Facility
      space_data.loc[(space_data["Country"] == 'Pacific Missile Range Facility'),"SpaceCenter"] = "Pacific Missile Range Facility"
      space_data.loc[(space_data["SpaceCenter"] == 'Pacific Missile Range Facility'),"State/Region"] = "Kauai"
      space_data.loc[(space_data["SpaceCenter"] == 'Pacific Missile Range Facility'),"Country"] = "USA"

      #Stargazer, Base Aerea de Gando, Gran Canaria---------(dropping the two rows)
      #space_data.loc[(space_data["Country"] == 'Gran Canaria'),"State/Region"] = "Gran Canaria"
      #space_data.loc[(space_data["State/Region"] == 'Gran Canaria'),"Country"] = "USA"
      space_data=space_data[(space_data.Country != "Gran Canaria")]

      #K-407 Submarine, Barents Sea Launch Area, Barents Sea
      space_data.loc[(space_data["Country"] == 'Barents Sea'),"State/Region"] = "Barents Sea"
      space_data.loc[(space_data["Country"] == 'Barents Sea'),"Country"] = "Russia"

      #Sea Launch - LP Odyssey, Kiritimati Launch Area, Pacific Ocean
      space_data.loc[(space_data["Country"] == 'Pacific Ocean'),"State/Region"] = "Pacific Ocean"
      space_data.loc[(space_data["State/Region"] == 'Pacific Ocean'),"Country"] = "Kiritimati"

      space_data.loc[(space_data["Country"] == "Kazakhstan"),"Country"] = "Russia"

      #fill empty values with NaN
      space_data['State/Region'] = fill_empty_with_NaN(space_data,'State/Region','')
    
      return space_data


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