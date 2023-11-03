import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit_elements import elements, mui, html, nivo, dashboard
import plotly.graph_objects as go

############################### Design Elements ###########################################################################################


st.set_page_config(layout='wide')

custom_css = """
<style>
    body {
        background-color: #F0E68C; /* Replace with your desired background color */
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<link rel="stylesheet.css" type="text/css" href="styles.css">', unsafe_allow_html=True)

############################### Imports ###################################################################################################

def import_url():
  
  gamned_logo_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Logo_G_Gamned_red_baseline.jpg'
  
  objective_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/format_table_last.csv'
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
  
  objective_list = ['branding display', 'branding video', 'consideration', 'conversion']
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

      
    if input_obj == 'branding display':
        df_rating.loc[df_rating['branding video'] == 0, 'branding'] += 10
        df_heatmap = df_rating[['channel', 'formats', 'branding']]
        df_heatmap = df_heatmap.sort_values(by='branding', ascending=False)

    elif input_obj == 'branding video':
        df_rating.loc[df_rating['branding video'] == 1, 'branding'] += 10
        df_heatmap = df_rating[['channel', 'formats', 'branding']]
        df_heatmap = df_heatmap.sort_values(by='branding', ascending=False)

    else:
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

if selected_objective == 'branding display' or selected_objective == 'branding video':
    selected_objective = 'branding'

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
  format_rating['channel'] = format_rating['channel'].replace('in game advertising', 'IGA')
  format_rating['format'] = format_rating['channel'] + '\n - ' + format_rating['formats']
  format_rating = format_rating[['channel', 'formats', 'format', selected_objective]]
  min_format = full_format_rating[selected_objective].min()
  max_format = full_format_rating[selected_objective].max()
  format_rating['norm'] = (format_rating[selected_objective] - min_format) / (max_format - min_format)*100
  format_rating['norm'] = format_rating['norm'].astype(float).round(0)
  #format_rating2 = format_rating.copy()
  #format_rating2['norm'] = format_rating2['norm'].apply(lambda x: x**2)
  #format_rating2['norm'] = format_rating2['norm'].astype(float).round(2)
  #format_rating['norm'] = format_rating['norm'].apply(round_5)
  #format_rating['mapped_colors'] = format_rating['norm'].map(color_dictionary)
  format_rating = format_rating.reset_index()
  format_rating = format_rating.drop(['index'], axis=1)
  
  return format_rating

format_rating = format_rating(df_rating3)

# Format rating is the component for the heatmap

############################################## Adding Price Rating ##############################################################


df_price = df_objective[['formats', 'price']]
df_price['price'] = df_price['price'] * 3



format_pricing = format_rating.copy()
format_pricing = pd.merge(format_pricing, df_price, on='formats')


format_pricing[selected_objective] = format_pricing[selected_objective] + format_pricing['price']

dropout = ['format', 'norm', 'price']
format_pricing = format_pricing.drop(columns=dropout)
format_pricing = format_pricing.sort_values(by=selected_objective, ascending=False)






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


##########################################  Budget Creation #############################################################################

def cost_norm(cost_rating):
  cost_rating = cost_rating.drop([selected_objective], axis=1)
  cost_rating = cost_rating.sort_values(by='average', ascending=False)
  cost_rating = cost_rating.reset_index(drop=True)
  cost_rating_std = cost_rating['average'].std()
  cost_rating_mean = cost_rating['average'].mean()
  cost_rating['norm'] = (cost_rating['average'] - cost_rating_mean) / cost_rating_std
  df_price_rating = cost_rating.copy()
  return cost_rating, df_price_rating

cost_rating, df_price_rating = cost_norm(cost_rating)


if channel_number == 0:
  if input_budget < 5001 and selected_objective == 'consideration':
    disp_allow = input_budget - 500
    budget_lib1 = {
      'channel': ['display', 'search'],
      'allowance': [disp_allow, 500]
    }
    df_allowance = pd.DataFrame(budget_lib1)
    
  elif input_budget < 5001:
    df_selection = cost_rating.head(1)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)
    
  elif input_budget < 10001 and input_budget > 5000:
    df_selection = cost_rating.head(2)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)

  elif input_budget < 15001 and input_budget > 10000:
    #df_selection = cost_rating[cost_rating['norm'] > threshold]
    df_selection = cost_rating.head(3)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)


  elif input_budget < 20001 and input_budget > 15000:
    #df_selection = cost_rating[cost_rating['norm'] > threshold]
    df_selection = cost_rating.head(4)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)

  elif input_budget < 25001 and input_budget > 20000:
    #df_selection = cost_rating[cost_rating['norm'] > threshold]
    df_selection = cost_rating.head(5)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)

  else:
    df_selection = cost_rating.head(6)
    df_budget = df_selection.copy()
    average_max = df_budget['average'].max()
    average_min = df_budget['average'].min()
    average_diff = average_max - average_min
    df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
    df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
    df_budget['allowance'] = input_budget * df_budget['distribution']
    columns_to_drop = ['average', 'norm', 'distribution']
    df_allowance = df_budget.drop(columns=columns_to_drop)
    
else:
  df_selection = cost_rating.head(channel_number)
  df_budget = df_selection.copy()
  average_max = df_budget['average'].max()
  average_min = df_budget['average'].min()
  average_diff = average_max - average_min
  df_budget['distribution'] = df_budget['average'] / df_budget['average'].sum()
  df_budget['distribution'] = df_budget['distribution'].apply(lambda x: round(x, 2))
  df_budget['allowance'] = input_budget * df_budget['distribution']
  columns_to_drop = ['average', 'norm', 'distribution']
  df_allowance = df_budget.drop(columns=columns_to_drop)
 

##########################################  Dashboard Content #######################################################################
#with open('styles.css') as f:
  #st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

######################################### heatmap ###################################################################################


def formatting_heatmap(format_rating, selected_objective):

    format_rating = format_rating.drop('format', axis=1)
    format_rating['channel'] = format_rating['channel'].str.upper()
    format_rating['formats'] = format_rating['formats'].str.title()
    format_rating['format'] = format_rating['channel'] + '\n - ' + format_rating['formats']
    top_format = format_rating.head(36)
    min_top_format = top_format['norm'].min()
    max_top_format = top_format['norm'].max()
    top_format = top_format.drop(selected_objective, axis=1)
    top_format['norm'] = (((top_format['norm'] - min_top_format) / (max_top_format - min_top_format)) * 100).round(0)
    top_format = top_format.sample(frac=1)
    return top_format

top_format = formatting_heatmap(format_rating, selected_objective)


def heatmap_data(top_format):
    
    top_format['format'] = top_format['format'].str.title()
    labels = top_format['format'].tolist()
    scores = top_format['norm'].to_numpy()
    scores_matrix = scores.reshape(6, 6)
    return labels, scores_matrix

labels, scores_matrix = heatmap_data(top_format)

    

# Sample data
#labels = [f"Label {i+1}" for i in range(48)]  # 8 columns x 6 rows = 48 labels
#scores = np.random.randint(0, 101, size=48)  # Generate random scores from 0 to 100

# Reshape scores into a 6x8 grid for the heatmap
#scores_matrix = scores.reshape(6, 8)

# Define a custom color scale with more shades of red and yellow
custom_color_scale = [
    [0, 'rgb(255, 255, 102)'],    # Light yellow
    [0.1, 'rgb(255, 255, 0)'],    # Yellow
    [0.2, 'rgb(255, 220, 0)'],    # Yellow with a hint of orange
    [0.4, 'rgb(255, 190, 0)'],    # Darker yellow
    [0.6, 'rgb(255, 140, 0)'],    # Light red-orange
    [0.7, 'rgb(255, 85, 0)'],     # Red-orange
    [0.8, 'rgb(255, 51, 0)'],     # Red
    [1, 'rgb(204, 0, 0)']         # Dark red
]

# Create a custom heatmap using Plotly with 8 columns and 6 rows
fig = go.Figure()

# Add the heatmap trace with the custom color scale
fig.add_trace(go.Heatmap(
    z=scores_matrix,
    colorscale=custom_color_scale,  # Use the custom color scale
    hoverongaps=False,
    showscale=False,  # Hide the color scale
    hovertemplate='%{z:.2f}<extra></extra>',  # Customize hover tooltip
))

# Add labels as annotations in the heatmap squares
for i, label in enumerate(labels):
    row = i // 6
    col = i % 6
    fig.add_annotation(
        text=label,
        x=col,
        y=row,
        xref='x',
        yref='y',
        showarrow=False,
        font=dict(size=10, color='black'),
        align='center'
    )

# Remove the axis labels and lines
fig.update_xaxes(showline=False, showticklabels=False)
fig.update_yaxes(showline=False, showticklabels=False)

fig.update_layout(
    width=800,  # Adjust the width as needed
    height=700,  # Adjust the height for 6 rows
    
    hovermode='closest',
)


# Display the Plotly figure in Streamlit with full width
st.plotly_chart(fig, use_container_width=True)



#################################################################################################################################

if df_allowance.shape[1] == 3:
    df_allowance = df_allowance.drop(selected_region, axis=1)


def pie_data(df_allowance):

  df_pie_chart = df_allowance.copy()
  df_pie_chart['allowance'] = df_pie_chart['allowance'].astype(int)
  df_pie_chart['channel'] = df_pie_chart['channel'].str.title()
  #df_pie_chart['channel'] = df_pie_chart['channel'].str.replace('In Game Advertising', 'In Game Advertising')
  return df_pie_chart
  
df_pie_chart = pie_data(df_allowance)

df_allow_table = df_pie_chart.copy()

new_cols = ['Channel', 'Budget']

df_allow_table.columns = new_cols




col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.metric('Top Channel', top_channel)

    st.metric('Budget', 'Youtube')

with col2:
    with elements("table"):
        with mui.Paper(eleveation=3, variant='outlined'):
            with mui.Box(sx={"height": 200}):
                st.write('Annoying')
        


with col3:
  
  with elements("pie_chart"):

      with mui.Paper(elevation=3, variant='outlined'):
          mui.Typography('Budget Analysis', variant='body2', padding='10px')

          with mui.Paper(elevation=3, variant='outlined', square=True):

  
              pie_chart_data = []
              
              for _, row in df_pie_chart.iterrows():
                allowance = {
                  'id': row['channel'],
                  'Label': row['channel'],
                  'value': row['allowance']
                }
                pie_chart_data.append(allowance)
          
              with mui.Box(sx={"height": 200}):
                        nivo.Pie(
                          data=pie_chart_data,
                          innerRadius=0.5,
                          cornerRadius=0,
                          padAngle=1,  
                          margin={'top': 30, 'right': 100, 'bottom': 30, 'left': 100},
                          theme={
                            "background": "#FFFFFF",
                            "textColor": "#31333F",
                            "tooltip": {
                                "container": {
                                    "background": "#FFFFFF",
                                    "color": "#31333F",
                                    }
                                }
                            }
                        )

     

################################################################################################################


with elements("properties"):

    # You can add properties to elements with named parameters.
    #
    # To find all available parameters for a given element, you can
    # refer to its related documentation on mui.com for MUI widgets,
    # on https://microsoft.github.io/monaco-editor/ for Monaco editor,
    # and so on.
    #
    # <Paper elevation={3} variant="outlined" square>
    #   <TextField label="My text input" defaultValue="Type here" variant="outlined" />
    # </Paper>

    with mui.Paper(elevation=3, variant="outlined", square=True):
        mui.TextField(
            label="My text input",
            defaultValue="Type here",
            variant="outlined",
           
            
              
        )


    # Streamlit Elements includes 45 dataviz components powered by Nivo.

        from streamlit_elements import nivo
    
        DATA = [
            { "taste": "fruity", "chardonay": 93, "carmenere": 61, "syrah": 114 },
            { "taste": "bitter", "chardonay": 91, "carmenere": 37, "syrah": 72 },
            { "taste": "heavy", "chardonay": 56, "carmenere": 95, "syrah": 99 },
            { "taste": "strong", "chardonay": 64, "carmenere": 90, "syrah": 30 },
            { "taste": "sunny", "chardonay": 119, "carmenere": 94, "syrah": 103 },
        ]
    
        with mui.Box(sx={"height": 500}):
            nivo.Radar(
                data=DATA,
                keys=[ "chardonay", "carmenere", "syrah" ],
                indexBy="taste",
                valueFormat=">-.2f",
                margin={ "top": 70, "right": 80, "bottom": 40, "left": 80 },
                borderColor={ "from": "color" },
                gridLabelOffset=36,
                dotSize=10,
                dotColor={ "theme": "background" },
                dotBorderWidth=2,
                motionConfig="wobbly",
                legends=[
                    {
                        "anchor": "top-left",
                        "direction": "column",
                        "translateX": -50,
                        "translateY": -40,
                        "itemWidth": 80,
                        "itemHeight": 20,
                        "itemTextColor": "#999",
                        "symbolSize": 12,
                        "symbolShape": "circle",
                        "effects": [
                            {
                                "on": "hover",
                                "style": {
                                    "itemTextColor": "#000"
                                }
                            }
                        ]
                    }
                ],
                theme={
                    "background": "#FFFFFF",
                    "textColor": "#31333F",
                    "tooltip": {
                        "container": {
                            "background": "#FFFFFF",
                            "color": "#31333F",
                        }
                    }
                }
            )

    # If you must pass a parameter which is also a Python keyword, you can append an
    # underscore to avoid a syntax error.
    #
    # <Collapse in />

    mui.Collapse(in_=True)

    # mui.collapse(in=True)
    # > Syntax error: 'in' is a Python keyword:




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



with elements("style_elements_css"):

    # For any other element, use the 'css' property.
    #
    # <div
    #   css={{
    #     backgroundColor: 'hotpink',
    #     '&:hover': {
    #         color: 'lightgreen'
    #     }
    #   }}
    # >
    #   This has a hotpink background
    # </div>

    html.div(
        "This has a hotpink background",
        css={
            "backgroundColor": "hotpink",
            "&:hover": {
                "color": "lightgreen"
            }
        }
    )


with elements("nested_children"):

    # You can nest children using multiple 'with' statements.
    #
    # <Paper>
    #   <Typography>
    #     <p>Hello world</p>
    #     <p>Goodbye world</p>
    #   </Typography>
    # </Paper>

    with mui.Paper:
        with mui.Typography:
            html.p("Hello world")
            html.p("Goodbye world")



st.dataframe(df_allowance)

import streamlit as st
import plotly.graph_objs as go

# Create a Plotly heatmap
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
fig = go.Figure(data=go.Heatmap(z=data))

# Define the Streamlit container with a white background
st.markdown(
    """
    <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);'>
        <h3>This is a container with a white background.</h3>
        <!-- Add more content inside the container if needed -->
    </div>
    """,
    unsafe_allow_html=True
)

# Display the Plotly heatmap inside the container
st.plotly_chart(fig)




