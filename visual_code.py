import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit_elements import elements, mui, html, nivo, dashboard
import plotly.graph_objects as go
import plotly.express as px

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
  
  gamned_logo_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Logo_Gamned_word_red.png'
  
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

col1, col2 = st.columns(2)

col1.image(gamned_logo_url, use_column_width=True)

col2.write(' ')
col2.write(' ')
col2.write(' ')
col2.write(' ')
col2.write(' ')
col2.subheader('Marketing Tool', divider='grey')

############################# Input Layer #######################################

def input_layer():

  target_list = ['b2c', 'b2b']
  target_df = pd.DataFrame(target_list)
  
  objective_list = ['branding display', 'branding video', 'consideration', 'conversion']
  objective_df = pd.DataFrame(objective_list)
  #objective_df.columns = ['0']
  objective_df[0] = objective_df[0].str.title()
  
  age_list = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'all']
  age_df = pd.DataFrame(age_list)
  
  country_list = ['None', 'GCC', 'KSA', 'UAE', 'KUWAIT', 'BAHRAIN', 'QATAR', 'OMAN']
  country_df = pd.DataFrame(country_list)
  
  excluded_channel_list = ['youtube', 'instagram', 'display', 'facebook', 'linkedin', 'search', 'snapchat', 'tiktok', 'native ads', 'twitter', 'twitch',
                      'in game advertising', 'amazon', 'audio', 'waze', 'dooh', 'connected tv']

  excluded_channel_list = [' '.join([word.capitalize() for word in item.split()]) for item in excluded_channel_list]
  
  box1, box2, box3, box4, box5, box6, box7 = st.columns(7)
  
  selected_objective = box1.selectbox('Objective', objective_df)
  selected_target = box2.selectbox('Target', target_df)
  selected_region = box3.selectbox('Region', country_df)
  excluded_channel = box4.multiselect('Channel to Exclude', excluded_channel_list)
  selected_age = box5.multiselect('Age', age_df)
  selected_age = ', '.join(selected_age)
  input_budget = box6.number_input('Budget $', value=0)
  channel_number = box7.number_input('Channel Number', value=0)
  search = st.checkbox('Include Search')
  

  return selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number, search

selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number, search = input_layer()

selected_objective = selected_objective.lower()

excluded_channel = [item.lower() for item in excluded_channel]

st.subheader(' ', divider='grey')


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
        df_rating.loc[df_rating['branding video'] == 1, 'branding'] -= 10
        df_heatmap = df_rating[['channel', 'formats', 'branding']]
        df_heatmap = df_heatmap.sort_values(by='branding', ascending=False)

    elif input_obj == 'branding video':
        df_rating.loc[df_rating['branding video'] == 1, 'branding'] += 10
        df_rating.loc[df_rating['branding video'] == 0, 'branding'] -= 10
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

##################################################### min price ##################################################################

min_price = {
    'channel': ['youtube', 'instagram', 'display', 'facebook', 'linkedin', 'search', 'snapchat', 'tiktok', 'native ads', 'twitter', 'twitch',
                      'in game advertising', 'amazon', 'audio', 'waze', 'dooh', 'connected tv'],
    'minimum': ['4000', '3000', '5000', '4000', '4000', '1000', '3000', '4000', '4000', '3000', '3000', '3000', '3000', '3000', '3000',
                 '3000', '3000']
}

min_price = pd.DataFrame(min_price)
min_price['minimum'] = min_price['minimum'].astype(int)


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
  format_rating['format'] = format_rating['channel'] + ' - ' + format_rating['formats']
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

def price_rating(df_objective, format_rating):

    df_objective['channel'] = df_objective['channel'].replace('in game advertising', 'IGA')
    df_objective['format'] = df_objective['channel'] + ' - ' + df_objective['formats']
    df_price = df_objective[['format', 'price']]
    df_price['price'] = df_price['price'] * 3
    
    
    format_pricing = format_rating.copy()
    format_pricing = format_pricing.merge(df_price, on='format', how='inner')
    format_pricing = format_pricing.drop_duplicates()
    
    format_pricing[selected_objective] = format_pricing[selected_objective] + format_pricing['price']
    
    dropout = ['format', 'norm', 'price']
    new_col = ['channel', 'formats', 'rating']
    format_pricing = format_pricing.drop(columns=dropout)
    format_pricing = format_pricing.rename(columns=dict(zip(format_pricing.columns, new_col)))
    format_pricing = format_pricing.sort_values(by='rating', ascending=False)
    return format_pricing

format_pricing = price_rating(df_objective, format_rating)

def round_up_with_infinity(x):
    if np.isinf(x):
        return x  # Leave infinite values unchanged
    else:
        return np.ceil(x)

format_pricing['rating'] = format_pricing['rating'].apply(round_up_with_infinity)


############################################# Building Budget ##################################################################


if channel_number == 0:

    if input_budget >= 10000 and input_budget < 15000:
        
        if search == True:
            budget = input_budget - 1000
            n_format = budget // 4000 + 1
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            
            search_row = {'channel': 'search', 'budget': 1000}
            budget_channel.loc[len(budget_channel.index)] = ['search', 1000]
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)

            

        else:

            n_format = input_budget // 4000 + 1
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = input_budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)

            
            

    elif input_budget >= 15000:
        
        if search == True:
            budget = input_budget - 2000
            n_format = budget // 4000 + 1
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            
            search_row = {'channel': 'search', 'budget': 1000}
            budget_channel.loc[len(budget_channel.index)] = ['search', 2000]
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)

            

        else:

            n_format = input_budget // 4000 + 1
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = input_budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)

            

    else:

        if search == True:
            
            budget = input_budget - 1000
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            n_format = 2
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel.loc[len(budget_channel.index)] = ['search', 1000]
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
    
            

        else:

            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            n_format = 2
            selected_format = format_pricing.head(n_format)
            unique_channel = selected_format['channel'].unique()
            unique_channel = pd.DataFrame({'channel': unique_channel}) 
            
            min_selection = unique_channel.merge(min_price, on='channel', how='inner')
            
            min_sum = min_selection['minimum'].sum()
            selected_format['budget'] = input_budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            
    
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
    
            

else:

    if input_budget < 15000:

        if search == True:
    
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            budget = input_budget - 1000
            uni_channels = set()
            consecutive_rows = []
        
            for index, row in format_pricing.iterrows():
                chan = row['channel']
                if chan not in uni_channels:
                    uni_channels.add(chan)
                    consecutive_rows.append(row.to_dict())
                if len(uni_channels) == channel_number:
                    break
        
            selected_format = pd.DataFrame(consecutive_rows)
            selected_format['budget'] = budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel.loc[len(budget_channel.index)] = ['search', 1000]
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
        
            

        else:
            
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            uni_channels = set()
            consecutive_rows = []
        
            for index, row in format_pricing.iterrows():
                chan = row['channel']
                if chan not in uni_channels:
                    uni_channels.add(chan)
                    consecutive_rows.append(row.to_dict())
                if len(uni_channels) == channel_number:
                    break
        
            selected_format = pd.DataFrame(consecutive_rows)
            selected_format['budget'] = input_budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
        
            

    else:

        if search == True:
    
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            budget = input_budget - 2000
            uni_channels = set()
            consecutive_rows = []
        
            for index, row in format_pricing.iterrows():
                chan = row['channel']
                if chan not in uni_channels:
                    uni_channels.add(chan)
                    consecutive_rows.append(row.to_dict())
                if len(uni_channels) == channel_number:
                    break
        
            selected_format = pd.DataFrame(consecutive_rows)
            selected_format['budget'] = budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel.loc[len(budget_channel.index)] = ['search', 2000]
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
        
            

        else:
            
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            uni_channels = set()
            consecutive_rows = []
        
            for index, row in format_pricing.iterrows():
                chan = row['channel']
                if chan not in uni_channels:
                    uni_channels.add(chan)
                    consecutive_rows.append(row.to_dict())
                if len(uni_channels) == channel_number:
                    break
        
            selected_format = pd.DataFrame(consecutive_rows)
            selected_format['budget'] = input_budget * selected_format['rating'] / (selected_format['rating'].sum())
            selected_format['budget'] = selected_format['budget'].round(0)
            budget_channel = selected_format.groupby('channel')['budget'].sum().reset_index()
            budget_channel = budget_channel.sort_values(by='budget', ascending=False)
        
        

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


##########################################  Bubble graph Data #######################################################################

format1 = df_objective.copy()
format1['channel'] = format1['channel'].replace('in game advertising', 'IGA')
format2 = selected_format.copy()
format3 = format_rating.copy()
format4 = format_rating.copy()

format1['unique'] = format1['channel'] + ' ' + format1['formats']
format2['unique'] = format2['channel'] + ' ' + format2['formats']
format3['unique'] = format3['channel'] + ' ' + format3['formats']
format4['unique'] = format4['channel'] + ' ' + format4['formats']

col_drop1 = ['branding', 'consideration', 'conversion', 'branding video']
col_drop2 = ['rating']
col_drop3 = ['format', 'norm']
col_drop4 = ['channel', 'formats', 'format', 'norm']

format1 = format1.drop(columns=col_drop1)
format2 = format2.drop(columns=col_drop2)
format3 = format3.drop(columns=col_drop3)
format3 = format3.head(3)
format4 = format4.drop(columns=col_drop4)


top_rating = format3.merge(format1, on='unique', how='inner')

top_rating = top_rating.drop_duplicates()
top_budget = format2.merge(format1, on='unique', how='inner')
top_budget = top_budget.drop_duplicates()
top_budget = top_budget.merge(format4, on='unique', how='inner')
col_drop1 = ['channel_y', 'formats_y', 'format']
col_drop2 = ['channel_y', 'formats_y', 'format']
top_rating = top_rating.drop(columns=col_drop1)
top_budget = top_budget.drop(columns=col_drop2)
new_val = [1000] * len(top_rating)
top_rating['budget'] = new_val
df_bubble = pd.concat([top_budget, top_rating])
df_bubble.reset_index(drop=True, inplace=True)
df_bubble = df_bubble.drop_duplicates(subset=['unique'])
drop_col = ['unique']
df_bubble = df_bubble.drop(columns=drop_col)
df_bubble[selected_objective] = df_bubble[selected_objective].apply(round_up_with_infinity)


######################################### heatmap ###################################################################################


def formatting_heatmap(format_rating, selected_objective):

    format_rating = format_rating.drop('format', axis=1)
    format_rating['channel'] = format_rating['channel'].str.upper()
    format_rating['formats'] = format_rating['formats'].str.title()
    format_rating['format'] = format_rating['channel'] + ' - ' + format_rating['formats']
    top_format = format_rating.head(42)
    min_top_format = top_format['norm'].min()
    max_top_format = top_format['norm'].max()
    top_format = top_format.drop(selected_objective, axis=1)
    top_format['norm'] = (((top_format['norm'] - min_top_format) / (max_top_format - min_top_format)) * 100).round(0)
    #top_format = top_format.sample(frac=1)
    top_format = top_format.sort_values(by='norm', ascending=True)
    return top_format

top_format = formatting_heatmap(format_rating, selected_objective)




def heatmap_data(top_format):
    
    top_format['format'] = top_format['format'].str.title()
    top_format['format'] = top_format['format'].replace('Twitter - Video Ads With Conversation Button', 'Twitter - Video Ads With Conv. Button')
    top_format['format'] = top_format['format'].replace('Twitter - Video Ads With Website Button', 'Twitter - Video Ads With Web. Button')
    labels = top_format['format'].tolist()
    scores = top_format['norm'].to_numpy()
    scores_matrix = scores.reshape(6, 7)
    scores_matrix = scores_matrix[::-1]
    return labels, scores_matrix

labels, scores_matrix = heatmap_data(top_format)

    

# Sample data
#labels = [f"Label {i+1}" for i in range(48)]  # 8 columns x 6 rows = 48 labels
#scores = np.random.randint(0, 101, size=48)  # Generate random scores from 0 to 100

# Reshape scores into a 6x8 grid for the heatmap
#scores_matrix = scores.reshape(6, 8)

# Define a custom color scale with more shades of red and yellow
custom_color_scale = [
    [0.00, 'rgb(255, 255, 102)'],    # Light yellow
    [0.05, 'rgb(255, 255, 0)'],      # Yellow
    [0.10, 'rgb(255, 220, 0)'],      # Yellow with a hint of orange
    [0.15, 'rgb(255, 200, 0)'],      # Another shade of yellow-orange
    [0.20, 'rgb(255, 190, 0)'],      # Darker yellow
    [0.25, 'rgb(255, 175, 0)'],      # Another shade of orange
    [0.30, 'rgb(255, 160, 0)'],      # Another shade of orange
    [0.35, 'rgb(255, 145, 0)'],      # Another shade of orange
    [0.40, 'rgb(255, 130, 0)'],      # Another shade of orange
    [0.45, 'rgb(255, 115, 0)'],      # Another shade of orange
    [0.50, 'rgb(255, 100, 0)'],      # Another shade of red-orange
    [0.55, 'rgb(255, 85, 0)'],       # Red-orange
    [0.60, 'rgb(255, 70, 0)'],       # Another shade of red-orange
    [0.65, 'rgb(255, 55, 0)'],       # Another shade of red-orange
    [0.70, 'rgb(255, 40, 0)'],       # Another shade of red-orange
    [0.75, 'rgb(255, 25, 0)'],       # Another shade of red-orange
    [0.80, 'rgb(255, 10, 0)'],       # Another shade of red-orange
    [0.85, 'rgb(204, 0, 0)'],        # Dark red
    [0.90, 'rgb(180, 0, 0)'],        # Another shade of dark red
    [0.95, 'rgb(160, 0, 0)'],        # Another shade of dark red
    [1.00, 'rgb(140, 0, 0)'],        # Another shade of dark red
]

# Create a custom heatmap using Plotly with 8 columns and 6 rows
fig = go.Figure()

# Add the heatmap trace with the custom color scale
fig.add_trace(go.Heatmap(
    z=scores_matrix,
    colorscale=custom_color_scale,  # Use the custom color scale
    hoverongaps=False,
    showscale=False,  # Hide the color scale
    hovertemplate='%{z:.2f}<extra></extra>',
    xgap=2,
    ygap=2# Customize hover tooltip
))


# Add labels as annotations in the heatmap squares
for i, label in enumerate(labels):
    row = i // 7
    col = i % 7
    
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
    width=750,  # Adjust the width as needed
    height=500,  # Adjust the height for 6 rows
    hovermode='closest',
    margin=dict(l=25, r=25, t=25, b=25),
    
    
)



# Display the Plotly figure in Streamlit with full width
st.plotly_chart(fig, use_container_width=True)



#################################################################################################################################

  
df_pie_chart = (budget_channel)

df_pie_chart['channel'] = df_pie_chart['channel'].str.title()
df_pie_chart['channel'] = df_pie_chart['channel'].replace('Iga', 'IGA')
df_pie_chart['budget'] = df_pie_chart['budget'].apply(lambda x: round(x, -1))

df_allow_table = df_pie_chart.copy()

new_cols = ['Channel', 'Budget']

df_allow_table.columns = new_cols

df_bubble.rename(columns={selected_objective: 'Rating'}, inplace=True)
df_bubble.rename(columns={'price': 'Price'}, inplace=True)
df_bubble['channel_x'] = df_bubble['channel_x'].str.title()
df_bubble['channel_x'] = df_bubble['channel_x'].replace('Iga', 'IGA')


if input_budget == 0: 
 st.write('Awaiting for budget...')

else:


 col1, col2 = st.columns(2)
 
 
 with col1:
  
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Budget Allocation")
   
    with elements("pie_chart"):
  
        
  
           
  
    
                pie_chart_data = []
                
                for _, row in df_pie_chart.iterrows():
                  allowance = {
                    'id': row['channel'],
                    'Label': row['channel'],
                    'value': row['budget']
                  }
                  pie_chart_data.append(allowance)
            
                with mui.Box(sx={"height": 400}):
                          nivo.Pie(
                            data=pie_chart_data,
                            innerRadius=0.5,
                            cornerRadius=0,
                            padAngle=1,  
                            margin={'top': 30, 'right': 100, 'bottom': 30, 'left': 100},
                            theme={
                              
                              "textColor": "#31333F",
                              "tooltip": {
                                  "container": {
                                      
                                      "color": "#31333F",
                                      }
                                  }
                              }
                          )
 
 
 with col2:
 
  
  
               st.write('Rating VS Price VS Budget')
 
               fig2 = px.scatter(df_bubble,
                                 x='Rating',
                                 y='Price',
                                 size='budget',
                                 color='channel_x',
                                 size_max=60,  # Increase the maximum bubble size
                                 log_x=True,
                                 text='channel_x',
                                 labels={'budget': 'Bubble Size'},  # Rename the legend label
                                 
                                 
                                )
 
               fig2.update_traces(textfont_color='black')
              
              # Set chart title and axis labels
               fig2.update_layout(
                   
                   showlegend=False,
                   width=600,
                   height=450,
                   margin=dict(l=25, r=25, t=50, b=25),
                   
               )
               
               # Display the Plotly figure in Streamlit
               
               st.plotly_chart(fig2)

     

################################################################################################################


st.subheader(' ', divider='grey')

details = st.checkbox('Show Details')

if details == True:
 selected_format['channel'] = selected_format['channel'].str.title()
 selected_format.columns = selected_format.columns.str.capitalize()
 st.dataframe(selected_format)









