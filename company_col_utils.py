import pandas as pd
import numpy as np
import os
import pycountry
from geopy.geocoders import Nominatim
import geopandas
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from data_cleaning_pre_processing import *

def location_split(x):
    """
    Split location
    :param x: columns
    :return: split value
    """
    t = x.split(',')
    if len(t) == 3:
        t.insert(2, "")
    return t

def calculate_company_success_launch_rate(space_data):
    """
    Calculate rocket launch success rate for each company with valid values
    :param space_data: dataframe to use
    :return: dataframe with success rate for each valid company
    """

    assert isinstance(space_data, pd.DataFrame)
    company_mission = space_data[['Company', 'MissionStatus']]

    company_mission = company_mission[company_mission['MissionStatus'] == "Success"]
    company_mission['success_count'] = company_mission.groupby(['Company'])['MissionStatus'].transform('count')
    company_mission = company_mission.drop_duplicates().reset_index(drop=True)
    company_mission_success_count = company_mission[['Company', 'success_count']]

    total_company_mission = space_data[['Company', 'MissionStatus']]
    total_company_mission["total_count"] = total_company_mission.groupby(['Company'])['MissionStatus'].transform(
        'count')
    total_company_mission = total_company_mission.drop_duplicates(subset=['Company']).drop(
        columns=['MissionStatus']).reset_index(drop=True)

    total_company_mission = pd.merge(total_company_mission, company_mission_success_count, on='Company',
                                     how='outer').fillna(0)
    total_company_mission["success_rate"] = total_company_mission["success_count"] / total_company_mission[
        "total_count"]
    return total_company_mission

def calculate_company_average_launch_cost(space_data):
    """
    Calculate average launch cost for each company. Exclude unreasonable values.
    :param space_data: dataframe to use
    :return: dataframe with success rate for each valid company
    """
    assert isinstance(space_data, pd.DataFrame)
    company_cost = space_data[['Company', 'MissionCost']]
    company_cost = company_cost[company_cost['MissionCost'] > 0]

    company_launch_count_cost = company_cost.copy()
    company_launch_count_cost['launch_count'] = company_launch_count_cost.groupby(['Company'])['MissionCost'].transform(
        'count')
    company_launch_count_cost = company_launch_count_cost.drop(columns=["MissionCost"])
    company_launch_count_cost = company_launch_count_cost.drop_duplicates().reset_index(drop=True)

    company_cost = company_cost.groupby("Company").sum()

    company_average_cost = company_cost.merge(company_launch_count_cost, on="Company", how="inner")
    company_average_cost["average_cost"] = company_average_cost["MissionCost"] / company_average_cost["launch_count"]
    return company_average_cost

def calculate_country_average_launch_cost(space_data):
    """
    Calculate average launch cost for each country.
    :param space_data: dataframe to use
    :return: dataframe with success rate for each valid country
    """
    assert isinstance(space_data, pd.DataFrame)
    country_cost = space_data[['Country', 'MissionCost']]
    country_cost = country_cost[country_cost['MissionCost'] > 0]

    country_launch_count_cost = country_cost.copy()
    country_launch_count_cost['launch_count'] = country_launch_count_cost.groupby(['Country'])['MissionCost'].transform(
        'count')
    country_launch_count_cost = country_launch_count_cost.drop(columns=["MissionCost"])
    country_launch_count_cost = country_launch_count_cost.drop_duplicates().reset_index(drop=True)

    country_cost = country_cost.groupby("Country").sum()

    country_average_cost = country_cost.merge(country_launch_count_cost, on="Country", how="inner")
    country_average_cost["average_cost"] = country_average_cost["MissionCost"] / country_average_cost["launch_count"]
    return country_average_cost

def alpha3code(column):
    """
    helper function to convert standard 3 code
    :param column: pd series
    :return: code columns
    """
    assert isinstance(column, pd.Series)
    CODE=[]
    for country in column:
        country = country.rstrip().lstrip()
        if country == "USA":
            country = "United States"
        elif country == "South Korea":
            country = "Korea, Republic of"
        elif country == "Russia":
            country = "Russian Federation"
        elif country == "North Korea":
            country = "Korea, Democratic People's Republic of"
        try:
            code=pycountry.countries.get(name=country).alpha_3
            CODE.append(code)
        except Exception as e:
            CODE.append('None')
    return CODE

def alpha2code(column):
    """
    helper function to convert standard 2 code
    :param column: pd series
    :return: code columns
    """
    assert isinstance(column, pd.Series)
    CODE=[]
    for country in column:
        country = country.rstrip().lstrip()
        if country == "USA":
            country = "United States"
        elif country == "South Korea":
            country = "Korea, Republic of"
        elif country == "Russia":
            country = "Russian Federation"
        elif country == "North Korea":
            country = "Korea, Democratic People's Republic of"
        try:
            code=pycountry.countries.get(name=country).alpha_2
            CODE.append(code)
        except Exception as e:
            CODE.append(None)
    return CODE


# function to get longitude and latitude data from country name
geolocator = Nominatim()
def geolocate(country):
    """
    calculate coords
    :param country: country name
    :return: location
    """
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan


def geolocate_col(column):
    """
    calculate location for a column
    :param column: pd series
    :return: coords as a new column
    """
    coords = []
    for code in column:
        coords.append(geolocate(code))
    return coords

def calculate_company_per_country(space_data):
    """
    Calculate number of company per country
    :return: dataframe with number of company in each country and location of the country for plotting
    """
    assert isinstance(space_data, pd.DataFrame)
    company_country = space_data[['Company', 'Country']]
    company_country = company_country.drop_duplicates('Company').reset_index(drop=True)
    company_country = company_country.dropna()
    company_country["counrty_count"] = company_country.groupby(['Country'])['Company'].transform('count')

    company_country["code"] = alpha3code(company_country.Country)
    company_country.dropna(inplace=True)

    company_country["coord"] = geolocate_col(company_country.code)
    company_country.dropna(inplace=True)

    # first let us merge geopandas data with our data
    # 'naturalearth_lowres' is geopandas datasets so we can use it directly
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    # rename the columns so that we can merge with our data
    world.columns = ['pop_est', 'continent', 'name', 'code', 'gdp_md_est', 'geometry']
    # then merge with our data
    merge = pd.merge(world, company_country, on='code', how="outer")
    # last thing we need to do is - merge again with our location data which contains each country’s latitude and longitude
    location = pd.read_csv(
        'https://raw.githubusercontent.com/melanieshi0120/COVID-19_global_time_series_panel_data/master/data/countries_latitude_longitude.csv')
    merge = merge.merge(location, on='name').sort_values(by='counrty_count', ascending=False).reset_index()

    merge[["counrty_count"]] = merge[['counrty_count']].fillna(value=0)
    return merge



def visualize_country_average_launch_cost(country_average_cost):
    """
    visualization
    :param country_average_cost:  calculate_country_average_launch_cost
    :return: plot
    """
    country_average_cost.sort_values(inplace=True, by='average_cost')
    ax = country_average_cost.plot(kind="barh", x='Country', y='average_cost', rot=0, figsize=(30, 10))

def visualize_company_average_launch_cost(company_average_cost):
    """
    visualization
    :param company_average_cost:  calculate_company_average_launch_cost
    :return: plot
    """
    company_average_cost.sort_values(inplace=True, by='average_cost')
    ax = company_average_cost.plot(kind="barh", x='Company', y='average_cost', rot=0, figsize=(30, 10))

def visualize_num_company_per_country(merge):
    """
    visualization
    :param company_average_cost:  calculate_company_per_country
    :return: plot
    """

    fig, ax = plt.subplots(1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    merge.plot(column='counrty_count', ax=ax, figsize=(25, 20), legend=True, cax=cax)
    plt.savefig("./fig.jpg", dpi=800)
