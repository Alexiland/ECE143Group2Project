import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



def month_country_count(df):
  '''
  Displays the number of launches each month of the top 4 countries in that month

  @param df: The relevant dataframe to be displayed
  @type df: pd.DataFrame
  '''
  
  assert isinstance(df, pd.DataFrame)

  month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

  df1 = df.groupby(['Month', 'Country']).size().reset_index().rename(columns={0:'Count'})
  df2 = df1
  x = []
  for i in range(3):
    df_sub = df2.sort_values('Count', ascending=False).drop_duplicates(['Month'])
    #Get the indexes of these 12 rows
    idx = df_sub.index.tolist()
    x = x + idx
    #Drop these 12 rows from df2
    df2 = df2.drop(idx)
    i += 1

  df1 = df1.loc[x]

  fig, ax = plt.subplots(figsize = (9,6))
  sns.barplot(data = df1, x = 'Month', y = 'Count', hue = 'Country', order = month_order, ax = ax).set(
    title = 'Top 5 Countries Launches Each Month'
    )



def launch_each_month(df):
  '''
  Creates a bar graph of the number of launches(y-axis) with respect to each month(x-axis)

  The bars are in descending order, with the month that has the most launches being the left-most bar

  @param df: Relevant dataframe to be plotted
  @type df: pd.DataFrame
  '''

  assert isinstance(df,pd.DataFrame)

  month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

  plt.figure(figsize=(7,5))
  ax = sns.countplot(x='Month', data=df, order=month_order, palette='Spectral')
  ax.axes.set_title('Launch Count for Each Month',fontsize=18)
  ax.set_xlabel('Month',fontsize=16)
  ax.set_ylabel('Count',fontsize=16)
  ax.tick_params(labelsize=11)
  plt.tight_layout()
  plt.show()



def monthly_cost_average(df):
    '''
    Create a plot that analyzes the average mission cost for each month

    @param df: The relevant dataframe
    @type df: pd.DataFrame
    '''

    assert isinstance(df, pd.DataFrame)

    month_to_cost = df.groupby('Month', as_index=False)['MissionCost'].mean()

    fig = px.bar(
      month_to_cost,
      x = 'Month',
      y = 'MissionCost',
      labels = {'Month': "Month", 'MissionCost': "Average Cost (In Millions)"},
      title = "Average Cost For Each Month"
    )
    fig.update_xaxes(categoryorder='array', categoryarray= ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                                                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    fig.show()



def yearly_cost_average(df):
    '''
    Create a plot that analyzes the average mission cost for each month

    @param df: The relevant dataframe
    @type df: pd.DataFrame
    '''

    assert isinstance(df, pd.DataFrame)

    year_to_cost = df.groupby('Year', as_index=False)['MissionCost'].mean()

    fig = px.bar(
      year_to_cost,
      x = 'Year',
      y = 'MissionCost',
      labels = {'Year': "Year", 'MissionCost': "Average Cost (In Millions)"},
      title = "Average Cost For Each Year"
    )
    fig.show()