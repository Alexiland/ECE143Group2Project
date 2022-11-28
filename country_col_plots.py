import pandas as pd
import plotly.express as px
from data_cleaning_pre_processing import *

def company_country_hist_plot(df):
  '''
  Returns a plotly fig which shows the histogram of number of companies in each country
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  data = df.groupby('Country',as_index=False)['Company'].nunique().sort_values(by=["Company"])
  fig = px.bar(data,x=data['Country'],y=data['Company'])
  fig.update_layout(title = "Number of Companies in each country",yaxis_title="Number of Comoanies")
  return fig


import pandas as pd
import plotly.express as px


def plot_hist(df, country, years, plot_title, y_axis_title):
    '''
    Returns a plotly fig object which is a histogram representing contribution of top-4 companies from the country over the years
    @param df: input dataframe
    @param country: country
    @param: years: list of years to be plotted
    @param plot_title: title of plot
    @param y_axis_title: title for y-axis
    @type df: pd.DataFrame
    @param country: str
    @param: years: list
    @param plot_title: str
    @param y_axis_title: str
    '''
    assert isinstance(df, pd.DataFrame)
    assert isinstance(country, str)
    assert isinstance(years, list)
    assert isinstance(plot_title, str)
    assert isinstance(y_axis_title, str)

    data = df[(df['Country'] == country) & (df['Year'].isin(years))]
    top_companies = list(pd.DataFrame(data['Company'].value_counts()).index)
    if (data['Company'].nunique() > 5):
        top_companies = list(pd.DataFrame(data['Company'].value_counts()[:4]).index)
    data = data[data['Company'].isin(top_companies)]
    fig = px.histogram(data, x="Year", color='Company')
    fig.update_layout(title=plot_title, yaxis_title=y_axis_title)
    return fig


def company_russia_plot(df):
    '''
    Returns a tuple of four plotly fig objects where each of the plots represents the below:
    => Russia Before 1990 : year-wise contribution of each company
    => Russia After 1990 : year-wise contribution of each company

    @param df: input dataframe
    @type df: pd.DataFrame

    '''
    assert isinstance(df, pd.DataFrame)

    years_1 = [x for x in range(1940, 1990)]
    years_2 = [x for x in range(1990, 2022)]

    fig_1 = plot_hist(df, 'Russia', years_1, "Russia Before 1990 : year-wise contribution of each company",
                      "Number of Missions")
    fig_2 = plot_hist(df, 'Russia', years_2, "Russia After 1990 : year-wise contribution of each company",
                      "Number of Missions")

    return fig_1, fig_2


def company_usa_plot(df):
    '''
    Returns a plotly fig pie-chart which explains the contribution of each company in USA
    @param df: input dataframe
    @type df: pd.DataFrame

    '''
    assert isinstance(df, pd.DataFrame)

    data = df[df['Country'] == 'USA'].groupby('Company', as_index=False).size()
    fig = px.pie(data, values='size', color='Company', names='Company')
    fig.update_traces(textposition='inside', textinfo='percent+label', title="Contribution of each company in USA")
    return fig

def country_missions_hist_plot(df):
  '''
  Returns a plotly fig which shows the histogram of number of missions (success and failure) by each country
  @param df: input dataframe
  @type df: pd.DataFrame
  '''

  assert isinstance(df,pd.DataFrame)

  fig = px.histogram(df, x = "Country", color = "MissionStatus")
  fig.update_layout(title = "Number of missions by each country")
  return fig

def trend_top_five_countries_plot(df):
  '''
  Returns a plotly fig line plot which shows the trend of number of missions of top five countries over the years
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  top_five_countries = list(pd.DataFrame(df['Country'].value_counts()[:5]).index)
  data=df[df['Country'].isin(top_five_countries)].groupby(['Year','Country'],as_index =False).size()
  fig=px.line(data,x='Year',y='size',color='Country')
  fig.update_layout(title="Year-wise trend of Top 5 countries",yaxis_title="Number of Missions")
  return fig

def trend_usa_and_russia_plot(df):
  '''
  Returns a plotly fig line plot which shows the trend of number of missions of USA and Russia over the years
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  data=df.groupby(['Year','Country','MissionStatus'],as_index =False).size()
  fig=px.line(data[data.Country.isin(['USA','Russia'])],x='Year',y='size',color='Country',animation_frame='MissionStatus')
  fig.update_layout(title="Year-wise trend of US and Russia",yaxis_title="Number of Missions")
  return fig

def add_iso_code_col(df):
  '''
  Returns the dataframe after adding a column with represents the corresponding ISO code for the country
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  country = ['USA', 'China', 'Japan', 'Israel', 'New Zealand',
            'Russia', 'Iran', 'France', 'India', 'North Korea',
            'Kiritimati', 'South Korea', 'Brazil', 'Kenya', 'Australia']
  iso_code = ['USA','CHN','JPN','ISR','NZL','RUS','IRN','FRA','IND','PRK','KIR','KOR','BRA','KEN','AUS']
  dict_country_codes = dict(zip(country,iso_code))
  df['ISOCode'] = pd.Series([dict_country_codes[x] for x in list(df['Country'])])
  return df


def total_missions_world_plot(df):
  '''
  Returns a plotly fig plot which shows the total number of missions of each country in a world heatmap
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  total_missions = df.groupby('Country',as_index=False).size()
  total_missions = add_iso_code_col(total_missions)

  fig = px.choropleth(total_missions, locations = "ISOCode", color="size",
            hover_name="Country",color_continuous_scale=px.colors.sequential.Sunsetdark)
  fig.update_layout(title="Total number of missions", coloraxis_colorbar_title_text = 'Number of Missions')

  return fig

def success_failure_rate_world_plot(df):
  '''
  Returns a plotly fig plot which shows the successs/failure rate of each country in a world heatmap
  @param df: input dataframe
  @type df: pd.DataFrame
  '''
  assert isinstance(df,pd.DataFrame)

  data = df.groupby(['Country','MissionStatus'],as_index=False).size()
  total_missions = df.groupby('Country',as_index=False).size()
  data['size'] = data['size'].astype(np.float64)

  for i in range(0,data.shape[0]):
    data['size'][i] = int(data['size'][i]) / int(total_missions.loc[total_missions['Country'] == data['Country'][i]]['size'])

  data = add_iso_code_col(data)
  fig=px.choropleth(data, locations = "ISOCode", animation_frame="MissionStatus", color="size",
            hover_name="Country",color_continuous_scale=px.colors.sequential.Sunsetdark)
  fig.update_layout(title="Success and Failure Rates", coloraxis_colorbar_title_text = 'Rate')
  return fig
