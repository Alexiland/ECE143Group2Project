import pandas as pd
import numpy as np
from helper_func import *

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
