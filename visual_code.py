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
  selected_age = sorted(selected_age)
  ################################### Sort age issue #####################################
  if selected_age == ['13-17', '25-34']:
   selected_age = ['13-17', '18-24', '25-34']
  elif selected_age == ['13-17', '35-44']:
   selected_age = ['13-17', '18-24', '25-34', '35-44']
  elif selected_age == ['13-17', '45-54']:
   selected_age = ['13-17', '18-24', '25-34', '35-44', '45-54']
  elif selected_age == ['13-17', '55-64']:
   selected_age = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64']
  elif selected_age == ['13-17', '65+']:
   selected_age = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
  elif selected_age == ['18-24', '35-44']:
   selected_age = ['18-24', '25-34', '35-44']
  elif selected_age == ['18-24', '45-54']:
   selected_age = ['18-24', '25-34', '35-44', '45-54']
  elif selected_age == ['18-24', '55-64']:
   selected_age = ['18-24', '25-34', '35-44', '45-54', '55-64']
  elif selected_age == ['18-24', '65+']:
   selected_age = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
  elif selected_age == ['25-34', '45-54']:
   selected_age = ['25-34', '35-44', '45-54']
  elif selected_age == ['25-34', '55-64']:
   selected_age = ['25-34', '35-44', '45-54', '55-64']
  elif selected_age == ['25-34', '65+']:
   selected_age = ['25-34', '35-44', '45-54', '55-64', '65+']
  elif selected_age == ['35-44', '55-64']:
   selected_age = ['35-44', '45-54', '55-64']
  elif selected_age == ['35-44', '65+']:
   selected_age = ['35-44', '45-54', '55-64', '65+']
  elif selected_age == ['45-54', '65+']:
    selected_age = ['45-54', '55-64', '65+']
  selected_age = ', '.join(selected_age)
  input_budget = box6.number_input('Budget $', value=0)
  channel_number = box7.number_input('Channel Number', value=0)
  input_search = st.slider('Search Allocation', 0, 3000, 0, 500)
  

  return selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number, input_search

selected_objective, selected_target, selected_region, excluded_channel, selected_age, input_budget, channel_number, input_search = input_layer()

selected_objective = selected_objective.lower()

decider = selected_objective

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
    col1 = ['instagram', 'facebook', 'linkedin', 'snapchat', 'youtube', 'tiktok', 'twitter', 'twitch', 'display', 'in game advertising',
           'connected tv', 'amazon', 'dooh']
    col2 = [8, 4.7, 0, 20, 0, 25, 7.8, 20, 14, 25, 5, 5, 0]
    col3 = [31, 21.5, 21.7, 38.8, 15, 33.8, 25.2, 21, 31, 30, 10, 10, 5]
    col4 = [30, 34.3, 40, 22.8, 20.7, 22.8, 26.6, 32, 27, 15, 15, 15, 10]
    col5 = [16, 19.3, 10, 13.8, 16.7, 13.8, 28.4, 17, 15, 5, 20, 30, 10]
    col6 = [8, 11.6, 20, 3.8, 11.9, 3.8, 8, 7, 7, 0, 30, 30, 10]
    col7 = [4, 4, 2.9, 0, 8.8, -5, 4, 3, 3, 0, 25, 20, 10]
    col8 = [-5, 2, 0, 0, 9, -5, 0, 0, 2, -10, 40, 30, 15]
    
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
    col30 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    
    

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
    age_table.iloc[0:, 1:] =  age_table.iloc[0:, 1:] / 2.5

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
        df_rating['id'] = df_rating['channel'] + '-' + df_rating['formats']
        df_rating = df_rating[df_rating['branding video'] == 0]
        df_heatmap = df_rating[['channel', 'formats', 'branding']]
        df_heatmap = df_heatmap.sort_values(by='branding', ascending=False)

    elif input_obj == 'branding video':
        
        df_rating['id'] = df_rating['channel'] + '-' + df_rating['formats']
        df_rating.loc[df_rating['id'] == 'instagram-Story', 'branding video'] = 1
        df_rating.loc[df_rating['id'] == 'facebook-Story', 'branding video'] = 1
        df_rating.loc[df_rating['id'] == 'snapchat-Story', 'branding video'] = 1
        df_rating.loc[df_rating['id'] == 'snapchat-Collection', 'branding video'] = 1
        df_rating = df_rating[df_rating['branding video'] == 1]
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

############################################# KnapSack Algo ####################################################################




############################################# Building Budget ##################################################################



if channel_number == 0:

    if input_budget >= 10000 and input_budget < 15000:
        
        #if search == True:
        if input_search > 0:
            #budget = input_budget - 1000
            budget = input_budget - input_search
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
            
            search_row = {'channel': 'search', 'budget': input_search}
            budget_channel.loc[len(budget_channel.index)] = ['search', input_search]
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
        
        if input_search > 0:
            budget = input_budget - input_search
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
            budget_channel.loc[len(budget_channel.index)] = ['search', input_search]
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

        if input_search > 0:
            
            budget = input_budget - input_search
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
            budget_channel.loc[len(budget_channel.index)] = ['search', input_search]
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

        if input_search > 0:
            #channel_number = channel_number - 1
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            budget = input_budget - input_search
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
            budget_channel.loc[len(budget_channel.index)] = ['search', input_search]
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

        if input_search > 0:
    
            format_pricing = format_pricing[format_pricing['channel'] != 'search']
            budget = input_budget - input_search
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
            budget_channel.loc[len(budget_channel.index)] = ['search', input_search]
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
#df_bubble = pd.concat([top_budget, top_rating])
df_bubble = top_budget.copy()
df_bubble.reset_index(drop=True, inplace=True)
df_bubble = df_bubble.drop_duplicates(subset=['unique'])
drop_col = ['unique']
df_bubble = df_bubble.drop(columns=drop_col)
df_bubble[selected_objective] = df_bubble[selected_objective].apply(round_up_with_infinity)



######################################### heatmap ###################################################################################


def formatting_heatmap(format_rating, selected_objective):

    
    if decider == 'branding video':
       format_rating = format_rating.drop('format', axis=1)
       format_rating['channel'] = format_rating['channel'].str.upper()
       format_rating['formats'] = format_rating['formats'].str.title()
       format_rating['format'] = format_rating['channel'] + ' - ' + format_rating['formats']
       if len(format_rating) >= 28:
        top_format = format_rating.head(28)
       else:
        default_value = np.nan 
        rows_to_add = 28 - len(format_rating)
        default_data = {'channel': [default_value] * rows_to_add,
                        'formats': [default_value] * rows_to_add,
                        'format': [default_value] * rows_to_add,
                        selected_objective: [default_value] * rows_to_add,
                        'norm': [default_value] * rows_to_add
                       }
        default_df = pd.DataFrame(default_data)
        top_format = pd.concat([format_rating, default_df], ignore_index=True)
       min_top_format = top_format['norm'].min()
       max_top_format = top_format['norm'].max()
       top_format = top_format.drop(selected_objective, axis=1)
       top_format['norm'] = (((top_format['norm'] - min_top_format) / (max_top_format - min_top_format)) * 100).round(0)
       top_format = top_format.sort_values(by='norm', ascending=True)
       return top_format
     
    else: 
  
       format_rating = format_rating.drop('format', axis=1)
       format_rating['channel'] = format_rating['channel'].str.upper()
       format_rating['formats'] = format_rating['formats'].str.title()
       format_rating['format'] = format_rating['channel'] + ' - ' + format_rating['formats']
       if len(format_rating) >= 42:
        top_format = format_rating.head(42)
       else:
        default_value = np.nan 
        rows_to_add = 42 - len(format_rating)
        default_data = {'channel': [default_value] * rows_to_add,
                        'formats': [default_value] * rows_to_add,
                        'format': [default_value] * rows_to_add,
                        selected_objective: [default_value] * rows_to_add,
                        'norm': [default_value] * rows_to_add
                       }
        default_df = pd.DataFrame(default_data)
        top_format = pd.concat([format_rating, default_df], ignore_index=True)
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
    
    return labels, scores_matrix




 ############################################# heatmap #######################################################################

top_format = top_format.sort_values(by='norm', ascending=False)
top_format['formats'] = top_format['formats'].replace('Video Ads With Conversation Button', 'Video Ads With Conv. Button')
top_format['formats'] = top_format['formats'].replace('Video Ads With Website Button', 'Video Ads With Web. Button')
top_format['formats'] = top_format['formats'].replace('Image Ads With Conversation Button', 'Image Ads With Conv. Button')
top_format['format'] = top_format['channel'] + '<br>' + top_format['formats']



def get_color(score):
    # You can define your own color mapping logic here
    if score == 0:
        return 'rgb(246, 247, 166)'
    elif score == np.nan:
        return 'rgb(255, 255, 255)'
    elif score < 0.05:
        return 'rgb(248, 250, 127)'
    elif score < 0.1:
        return 'rgb(245, 247, 77)'
    elif score < 0.15:
        return 'rgb(247, 239, 77)'
    elif score < 0.2:
        return 'rgb(247, 224, 77)'
    elif score < 0.25:
        return 'rgb(247, 210, 77)'
    elif score < 0.3:
        return 'rgb(247, 196, 77)'
    elif score < 0.35:
        return 'rgb(247, 185, 77)'
    elif score < 0.4:
        return 'rgb(247, 173, 77)'
    elif score < 0.45:
        return 'rgb(247, 165, 77)'
    elif score < 0.5:
        return 'rgb(247, 159, 77)'
    elif score < 0.55:
        return 'rgb(247, 148, 77)'
    elif score < 0.6:
        return 'rgb(247, 134, 77)'
    elif score < 0.65:
        return 'rgb(247, 125, 77)'
    elif score < 0.7:
        return 'rgb(247, 114, 77)'
    elif score < 0.75:  # Fixed threshold (was missing)
        return 'rgb(247, 105, 77)'
    elif score < 0.8:
        return 'rgb(247, 97, 77)'
    elif score < 0.85:
        return 'rgb(240, 29, 29)'
    elif score < 0.9:  # Fixed threshold (was missing)
        return 'rgb(214, 24, 24)'
    elif score < 0.95:  # Fixed threshold (was missing)
        return 'rgb(191, 21, 21)'
    elif score < 1:
        return 'rgb(179, 21, 21)'
    elif score == 1.0:
        return 'rgb(163, 20, 20)'
    else:
        return 'rgb(255, 255, 255, 0)'


def get_text_color(background_color):

 rgb_values = background_color.strip('rgb()').split(',')
 
 r, g, b = map(int, rgb_values)
 luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
 if luminance > 0.5:
  return 'black'
 else:
  return 'white'

if decider == 'branding video':

  index1 = [0, 7, 14, 21]
  index2 = [1, 8, 15, 22]
  index3 = [2, 9, 16, 23]
  index4 = [3, 10, 17, 24]
  index5 = [4, 11, 18, 25]
  index6 = [5, 12, 19, 26]
  index7 = [6, 13, 20, 27]

  heatmap1 = top_format.iloc[index1]
  heatmap2 = top_format.iloc[index2]
  heatmap3 = top_format.iloc[index3]
  heatmap4 = top_format.iloc[index4]
  heatmap5 = top_format.iloc[index5]
  heatmap6 = top_format.iloc[index6]
  heatmap7 = top_format.iloc[index7]

else:
 
  index1 = [0, 7, 14, 21, 28, 35]
  index2 = [1, 8, 15, 22, 29, 36]
  index3 = [2, 9, 16, 23, 30, 37]
  index4 = [3, 10, 17, 24, 31, 38]
  index5 = [4, 11, 18, 25, 32, 39]
  index6 = [5, 12, 19, 26, 33, 40]
  index7 = [6, 13, 20, 27, 34, 41]
  
  heatmap1 = top_format.iloc[index1]
  heatmap2 = top_format.iloc[index2]
  heatmap3 = top_format.iloc[index3]
  heatmap4 = top_format.iloc[index4]
  heatmap5 = top_format.iloc[index5]
  heatmap6 = top_format.iloc[index6]
  heatmap7 = top_format.iloc[index7]


# Define a function to map scores to colors



col10, col11, col12, col13, col14, col15, col16, col17, col18= st.columns([1, 2, 2, 2, 2, 2, 2, 2, 1])

with col11:

 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.4); 
             transition: box-shadow 0.3s ease-in-out;
             position: relative;
             
         }

         .heatmap-item::before {
            content: "";
            position: absolute;
            top: 0px;
            left: 0px;
            right: 0px;
            bottom: 0px;
            box-shadow: 0 0 1px rgba(0, 0, 0, 0.7) inset, 0 0 20px rgba(255, 255, 255, 0.7) inset; /* Add an inset shadow for 3D effect */
            border-radius: inherit; /* Inherit border radius from parent */
        }
         
         .heatmap-item:hover {
             box-shadow: 0 32px 64px rgba(0, 0, 0, 0.2);
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap1.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )

with col12:

 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap2.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )


with col13:
 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap3.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )


with col14:
 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap4.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )



with col15:
 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap5.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )


with col16:
 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap6.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )



with col17:
 with st.container():
     st.markdown(
         """
         <style>
         .heatmap-container {
             display: flex;
             flex-direction: column; /* Arrange squares vertically */
         }
 
         .heatmap-item {
             width: 150px;
             height: 75px;
             margin-bottom: 10px; /* Add margin at the bottom of each square */
             font-size: 10px;
             display: flex;
             align-items: center;
             justify-content: center;
             border-radius: 10px;
             
             box-shadow: 0 6px 10px 0 rgba(0, 0, 0, 0.2); /* Add a box shadow for 3D effect */
         }
         </style>
         """,
         unsafe_allow_html=True
     )
 
     for index, row in heatmap7.iterrows():
         name = row['channel']
         format = row['formats']
         score = row['norm'] / 100
         if score >= 0:
          color = get_color(score)
          text_color = get_text_color(color)
         else:
          continue
 
         # Use the 'st.markdown' to create colored boxes with shadows and labels
         st.markdown(
             f"""
             <div class="heatmap-item" style="background-color: {color}; text-align: center; font-size: 12px; color: {text_color};">
                 {name}<br>
                 {format}
             </div>
             """,
             unsafe_allow_html=True
         )

# Sample data
#labels = [f"Label {i+1}" for i in range(48)]  # 8 columns x 6 rows = 48 labels
#scores = np.random.randint(0, 101, size=48)  # Generate random scores from 0 to 100

# Reshape scores into a 6x8 grid for the heatmap
#scores_matrix = scores.reshape(6, 8)

# Define a custom color scale with more shades of red and yellow



#################################################################################################################################

st.divider()
  
df_pie_chart = (budget_channel)

df_pie_chart['channel'] = df_pie_chart['channel'].str.title()
df_pie_chart['channel'] = df_pie_chart['channel'].replace('Iga', 'IGA')
df_pie_chart['budget'] = df_pie_chart['budget'].apply(lambda x: round(x, -1))

df_allow_table = df_pie_chart.copy()

new_cols = ['Channel', 'Budget']

df_allow_table.columns = new_cols

df_bubble.rename(columns={selected_objective: 'Rating'}, inplace=True)
df_bubble.rename(columns={'price': 'Price'}, inplace=True)
df_bubble['Price'] = df_bubble['Price'] + np.round(np.random.rand(len(df_bubble)), 1)
df_bubble['Price'] = np.log(df_bubble['Price'])
df_bubble['channel_x'] = df_bubble['channel_x'].str.title()
df_bubble['channel_x'] = df_bubble['channel_x'].replace('Iga', 'IGA')
df_bubble['format'] = df_bubble['channel_x'] + '\n' + df_bubble['formats_x']



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
                                 text='format',
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


########################################################### Formatting data for heatmap ######################################
















