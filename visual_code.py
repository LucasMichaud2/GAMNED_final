import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from streamlit_elements import elements, mui, html, nivo

############################### Imports ###########################################

gamned_logo_url = 'https://raw.github.com/LucasMichaud2/GAMNED_final/main/Logo_G_Gamned_red_baseline.jpg'

############################### Design Elements ###################################

st.markdown('<link rel="stylesheet.css" type="text/css" href="styles.css">', unsafe_allow_html=True)

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
 
 
