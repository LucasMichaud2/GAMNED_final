import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit_elements import elements, mui, html, nivo, dashboard

############################### Design Elements ###########################################################################################

st.set_page_config(layout='wide')
st.markdown('<link rel="stylesheet.css" type="text/css" href="styles.css">', unsafe_allow_html=True)

############################### Imports ###################################################################################################

def import_url():
  
  gamned_logo_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Logo_G_Gamned_red_baseline.jpg'
  
  objective_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Objectives_updated-Table%201.csv'
  df_objective = pd.read_csv(objective_url)

  data_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/GAMNED_dataset_V2.2.csv'
  df_data = pd.read_csv(data_url)

  age_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Global_data-Table%201.csv'
  age_date = pd.read_csv(age_url)

  weighted_country_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/weighted_country.csv'
  weighted_country = pd.read_csv(weighted_country_url)

  return gamned_logo_url, df_objective, df_data, age_date, weighted_country


gamned_logo_url, df_objective, df_data, age_date, weighted_country = import_url()

############################## Title Layer #######################################

col1, col2, col3, col4 = st.columns(4)

col1.header('Marketing Tool')

############################# Input Layer #######################################

def input_layer():

  target_list = ['b2c', 'b2b']
  target_df = pd.DataFrame(target_list)
  
  objective_list = ['branding', 'consideration', 'conversion']
  objective_df = pd.DataFrame(objective_list)
  
  age_list = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'all']
  age_df = pd.DataFrame(age_list)
  
  country_list = ['None', 'GCC', 'KSA', 'UAE', 'KUWAIT', 'BAHRAIN', 'QATAR', 'OMAN']
  country_df = pd.DataFrame(country_list)
  
  excluded_channel_list = ['youtube', 'instagram', 'display', 'facebook', 'linkedin', 'search', 'snapchat', 'tiktok', 'native ads', 'twitter', 'twitch',
                      'in game advertising', 'amazon', 'audio', 'waze', 'dooh', 'connected tv']
  
  box1, box2, box3, box4, box5, box6, box7 = st.columns(7)
  
  selected_objective = box1.selectbox('Select Objective', objective_df)
  selected_target = box2.selectbox('Select target', target_df)
  selected_region = box3.selectbox('Select Region', country_df)
  excluded_channel = box4.multiselect('Channel to Exclude', excluded_channel_list)
  selected_age = box5.multiselect('Select an Age', age_df)
  selected_age = ', '.join(selected_age)
  input_budget = box6.number_input('Budget', value=0)
  channel_number = box7.number_input('Number of Channels', value=0)

  return selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number

selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number = input_layer()


############################## Class Import ##############################################################################################

class GAMNED_UAE:


  def __init__(self, data, rating):
    self.df_data = data[data['country'] == 'uae']
    self.df_rating = rating
    self.obj_list = ['branding', 'consideration', 'conversion']

  def get_age_data(self):
    column_names = ['channel', '13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', '13-17, 18-24', '13-17, 18-24, 25-34',
                   '13-17, 18-24, 25-34, 35-44',
                   '13-17, 18-24, 25-34, 35-44, 45-54', 
                   '13-17, 18-24, 25-34, 35-44, 45-54, 55-64', 
                   '13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+', 
                   'all',
                   '18-24, 25-34', 
                   '18-24, 25-34, 35-44', 
                   '18-24, 25-34, 35-44, 45-54', 
                   '18-24, 25-34, 35-44, 45-54, 55-64',
                   '18-24, 25-34, 35-44, 45-54, 55-64, 65+',
                   '25-34, 35-44',
                   '25-34, 35-44, 45-54',
                   '25-34, 35-44, 45-54, 55-64', 
                   '25-34, 35-44, 45-54, 55-64, 65+',
                   '35-44, 45-54', 
                   '35-44, 45-54, 55-64', 
                   '35-44, 45-54, 55-64, 65+', 
                   '45-54, 55-64', 
                   '45-54, 55-64, 65+',
                   '55-64, 65+',
                   '']
    col1 = ['instagram', 'facebook', 'linkedin', 'snapchat', 'youtube']
    col2 = [8, 4.7, 0, 20, 0]
    col3 = [31, 21.5, 21.7, 38.8, 15]
    col4 = [30, 34.3, 60, 22.8, 20.7]
    col5 = [16, 19.3, 10, 13.8, 16.7]
    col6 = [8, 11.6, 5.4, 3.8, 11.9]
    col7 = [4, 7.2, 2.9, 0, 8.8]
    col8 = [3, 5.6, 0, 0, 9]
    col9 = [x + y for x, y in zip(col2, col3)]
    col10 = [x + y for x, y in zip(col9, col4)]
    col11 = [x + y for x, y in zip(col10, col5)]
    col12 = [x + y for x, y in zip(col11, col6)]
    col13 = [x + y for x, y in zip(col12, col7)]
    col14 = [x + y for x, y in zip(col13, col8)]
    col15 = [x + y for x, y in zip(col3, col4)]
    col16 = [x + y for x, y in zip(col15, col5)]
    col17 = [x + y for x, y in zip(col6, col6)]
    col18 = [x + y for x, y in zip(col17, col7)]
    col19 = [x + y for x, y in zip(col18, col8)]
    col20 = [x + y for x, y in zip(col4, col5)]
    col21 = [x + y for x, y in zip(col20, col6)]
    col22 = [x + y for x, y in zip(col21, col7)]
    col23 = [x + y for x, y in zip(col22, col8)]
    col24 = [x + y for x, y in zip(col5, col6)]
    col25 = [x + y for x, y in zip(col24, col7)]
    col26 = [x + y for x, y in zip(col25, col8)]
    col27 = [x + y for x, y in zip(col6, col7)]
    col28 = [x + y for x, y in zip(col27, col8)]
    col29 = [x + y for x, y in zip(col7, col8)]
    col30 = [0, 0, 0, 0, 0]
    
    
    

    df_age = pd.DataFrame(col1, columns = ['channel'])
    df_age['13-17'] = col2
    df_age['18-24'] = col3
    df_age['25-34'] = col4
    df_age['35-44'] = col5
    df_age['45-54'] = col6
    df_age['55-64'] = col7
    df_age['65+'] = col8
    df_age['13-17, 18-24'] = col9
    df_age['13-17, 18-24, 25-34'] = col10
    df_age['13-17, 18-24, 25-34, 35-44'] = col11
    df_age['13-17, 18-24, 25-34, 35-44, 45-54'] = col12
    df_age['13-17, 18-24, 25-34, 35-44, 45-54, 55-64'] = col13
    df_age['13-17, 18-24, 25-34, 35-44, 45-54, 55-64, 65+'] = col14
    df_age['all'] = col14
    df_age['18-24, 25-34'] = col15
    df_age['18-24, 25-34, 35-44'] = col16
    df_age['18-24, 25-34, 35-44, 45-54'] = col17
    df_age['18-24, 25-34, 35-44, 45-54, 55-64'] = col18
    df_age['18-24, 25-34, 35-44, 45-54, 55-64, 65+'] = col19
    df_age['25-34, 35-44'] = col20
    df_age['25-34, 35-44, 45-54'] = col21
    df_age['25-34, 35-44, 45-54, 55-64'] = col22
    df_age['25-34, 35-44, 45-54, 55-64, 65+'] = col23
    df_age['35-44, 45-54'] = col24
    df_age['35-44, 45-54, 55-64'] = col25
    df_age['35-44, 45-54, 55-64, 65+'] = col26
    df_age['45-54, 55-64'] = col27
    df_age['45-54, 55-64, 65+'] = col28
    df_age['55-64, 65+'] = col29
    df_age[''] = col30
    
    return df_age



  def uniform_channel(self):
    # make sure we have one exhaustive channel list
    channel_list = pd.concat([self.df_data['channel'], self.df_rating['channel']], axis=0)
    channel_list = channel_list.unique().tolist()
    return channel_list


  def get_mean_rating(self):
    # Get the rating df and output the mean value for each channel
    # The df is all the channels and mean rating in relation to (branding,
    # consideration and conversion)
    mean_ratings = self.df_rating.drop('formats', axis=1)
    mean_ratings = mean_ratings.groupby('channel').mean().reset_index()
    return mean_ratings

  def get_data_freq(self):
    # get the freq of channel used in data for each objectives
    df_freq = self.df_data[['objectif', 'channel']]
    stack_list = []
    for i in self.obj_list:
      temp_df = df_freq[df_freq['objectif'] == i]
      stack_list.append(temp_df['channel'].value_counts())
    df_freq = pd.DataFrame(stack_list)
    df_freq = df_freq.T
    df_freq.columns = self.obj_list
    df_freq = df_freq / df_freq.sum()
    df_freq = df_freq*10
    df_freq = df_freq.reset_index()
    df_freq.rename(columns={'index': 'channel'}, inplace=True)
    return df_freq

  def get_channel_rating(self, input_age, df_age, df_freq, df_rating):
    # Get the final rating of channel and formats depending on age group
    # and objective

    age_column = df_age[input_age].tolist()
    age_channel = df_age['channel'].tolist()

    age_dic = {
        'channel': age_channel,
        'branding': age_column,
        'consideration': age_column,
        'conversion': age_column
    }
    age_table = pd.DataFrame(age_dic)
    age_table.iloc[0:, 1:] =  age_table.iloc[0:, 1:] / 10

    temp1 = pd.concat([df_freq, age_table], axis=0)
    temp1 = pd.concat([temp1, df_rating], axis=0)

    df_channel_rating = temp1.groupby('channel').sum()
    #df_channel_rating.columns = ['channel', 'branding', 'consideration', 'converison']
    df_channel_rating = df_channel_rating.reset_index()

    return df_channel_rating


  def get_format_rating(self, channel_rating):
    # combine format and channel rating

    for index, row in channel_rating.iterrows():

      a_value = row['channel']
      self.df_rating.loc[self.df_rating['channel'] == a_value, self.obj_list] += row[self.obj_list]

    return self.df_rating


  def get_objective(self, input_obj, df_rating):

    df_heatmap = df_rating[['channel', 'formats', input_obj]]
    df_heatmap = df_heatmap.sort_values(by=input_obj, ascending=False)
    return df_heatmap


  def get_target(self):

    target_dic = {
        'channel': ['linkedin', 'search', 'video', 'native ads'],
        'branding': [10, 10, 10, 10],
        'consideration': [10, 10, 10, 10],
        'conversion': [10, 10, 10, 10]
    }

    df_target = pd.DataFrame(target_dic)

    return df_target

  def add_target(self, df_target, channel_rating):

      total_rating = pd.concat([channel_rating, df_target], axis=0)
      total_rating = total_rating.groupby('channel').sum()

      return total_rating


################################ Applying Class ###################################################################################

gamned_class = GAMNED_UAE(df_data, df_objective)

def apply_class():
  
  df_age = gamned_class.get_age_data()
  df_freq = gamned_class.get_data_freq()
  df_rating = gamned_class.get_mean_rating()
  df_rating1 = gamned_class.get_channel_rating(selected_age, df_age, df_freq, df_rating)
  if selected_target == 'b2b':
    df_b2b = gamned_class.get_target()
    df_rating1 = gamned_class.add_target(df_b2b, df_rating1)
    df_rating1 = df_rating1.reset_index()
  df_rating2 = gamned_class.get_format_rating(df_rating1)
  df_rating3 = gamned_class.get_objective(selected_objective, df_rating2)
  df_rating3 = df_rating3[~df_rating3['channel'].isin(excluded_channel)]
  df_rating3 = df_rating3.reset_index(drop=True)

  return df_rating3

df_rating3 = apply_class()

########################################## Country Ratings #######################################################################

def country_rating(df_rating3):

  if selected_region != 'None':

    df_region = weighted_country[['channel', selected_region]]
    region_max = df_region[selected_region].max()
    region_min = df_region[selected_region].min()
    df_region[selected_region] = ((df_region[selected_region] - region_min) / (region_max - region_min))*10
    df_rating3 = df_rating3.merge(df_region, on='channel', how='left')
    df_rating3[selected_objective] = df_rating3[selected_objective] + df_rating3[selected_region]
    df_rating3 = df_rating3.sort_values(by=selected_objective, ascending=False)

  return df_rating3

df_rating3 = country_rating(df_rating3)


################################################ Format Ratings #################################################################

def format_rating(df_rating3):

  full_format_rating = df_rating3.copy()
  format_rating = df_rating3.copy()
  format_rating['format'] = format_rating['channel'] + '\n' + format_rating['formats']
  format_rating = format_rating[['channel', 'formats', 'format', selected_objective]]
  min_format = full_format_rating[selected_objective].min()
  max_format = full_format_rating[selected_objective].max()
  format_rating['norm'] = (format_rating[selected_objective] - min_format) / (max_format - min_format)*100
  format_rating['norm'] = format_rating['norm'].astype(float).round(2)
  #format_rating2 = format_rating.copy()
  #format_rating2['norm'] = format_rating2['norm'].apply(lambda x: x**2)
  #format_rating2['norm'] = format_rating2['norm'].astype(float).round(2)
  #format_rating['norm'] = format_rating['norm'].apply(round_5)
  #format_rating['mapped_colors'] = format_rating['norm'].map(color_dictionary)
  format_rating = format_rating.reset_index()
  format_rating = format_rating.drop(['index'], axis=1)
  
  return format_rating

format_rating = format_rating(df_rating3)


############################################## Getting the Channel rating by agg formats ########################################

def agg_channel_rating(df_rating3):
  
  channel_count = pd.DataFrame(df_rating3.groupby('channel')['formats'].count())
  channel_count = channel_count.reset_index()
  col_names = ['channel', 'count']
  channel_count.columns = col_names
  
  agg_rating = df_rating3.drop(['formats'], axis=1)
  agg_rating1 = agg_rating.groupby('channel').sum()
  agg_rating1 = agg_rating1.reset_index()
  agg_rating2 = agg_rating1.sort_values(by='channel')
  channel_count2 = channel_count.sort_values(by='channel')
  agg_rating2['average'] = agg_rating2[selected_objective] / channel_count2['count']
  agg_rating3 = agg_rating2.sort_values(by='average', ascending=False)
  
  cost_rating = agg_rating3.copy()
  agg_rating4 = agg_rating3.copy()
  agg_rating4 = agg_rating4.reset_index(drop=True)
  cost_rating = cost_rating.reset_index(drop=True)
  
  agg_rating_min = agg_rating3['average'].min()
  agg_rating_max = agg_rating3['average'].max()
  agg_rating3['average'] = ((agg_rating3['average'] - agg_rating_min) / (agg_rating_max - agg_rating_min))*100
  output_rating = agg_rating3.copy()

  return cost_rating, agg_rating4, output_rating

cost_rating, agg_rating4, output_rating = agg_channel_rating(df_rating3)


############################################# Gettign the top Channel ###################################################################

def top_channel(agg_rating4):

  top_channel = agg_rating4.at[0, 'channel']
  top_channel = top_channel.title()
  return top_channel

top_channel = top_channel(agg_rating4)


##########################################  Dashboard Content ########################################################################

with elements('dashboard'):

  layout = [
    dashboard.Item('top_channel', 0, 0, 2, 2),
  ]

  with dashboard.Grid(layout):

    with mui.Paper(key='top_channel', style={'background-color': 'lightblue', 'padding': '10px'}):
      with mui.Div(style={'textAlign': 'center'}):
        mui.H3('Top Format', style={'color': 'black'})
      
     

################################################################################################################

with elements('Title Layer'):

 mui.Typography("Hello World")



with elements("style_mui_sx"):

    # For Material UI elements, use the 'sx' property.
    #
    # <Box
    #   sx={{
    #     bgcolor: 'background.paper',
    #     boxShadow: 1,
    #     borderRadius: 2,
    #     p: 2,
    #     minWidth: 300,
    #   }}
    # >
    #   Some text in a styled box
    # </Box>

    mui.Box(
        "Some text in a styled box",
        sx={
            "bgcolor": "background.paper",
            "boxShadow": 1,
            "borderRadius": 2,
            "p": 2,
            "minWidth": 300,
        }
    )



with elements("dashboard"):

    # You can create a draggable and resizable dashboard using
    # any element available in Streamlit Elements.

    from streamlit_elements import dashboard

    # First, build a default layout for every element you want to include in your dashboard

    layout = [
        # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
        dashboard.Item("first_item", 0, 0, 2, 2),
        dashboard.Item("second_item", 2, 0, 2, 2, isDraggable=False, moved=False),
        dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
    ]

    # Next, create a dashboard layout using the 'with' syntax. It takes the layout
    # as first parameter, plus additional properties you can find in the GitHub links below.

    with dashboard.Grid(layout):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")

    # If you want to retrieve updated layout values as the user move or resize dashboard items,
    # you can pass a callback to the onLayoutChange event parameter.

    def handle_layout_change(updated_layout):
        # You can save the layout in a file, or do anything you want with it.
        # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        print(updated_layout)

    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")
 
 
