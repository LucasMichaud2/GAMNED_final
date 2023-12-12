# Sample data
labels = [f"Label {i+1}" for i in range(36)]
scores = np.random.rand(36)

# Reshape scores into a 6x6 grid for the heatmap
scores_matrix = scores.reshape(6, 6)

# Create a custom heatmap using Plotly
fig = go.Figure()

# Add the heatmap trace
fig.add_trace(go.Heatmap(
    z=scores_matrix,
    colorscale='Viridis',  # You can choose different color scales
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
        font=dict(size=10)
    )

fig.update_xaxes(side="top")
fig.update_layout(
    width=800,
    height=800,
    title='Custom Heatmap with Hover Effect',
    xaxis_title='X-axis',
    yaxis_title='Y-axis',
    hovermode='closest',
)

# Add a custom JavaScript snippet to apply hover effect
hover_code = """
<script>
    var cells = document.getElementsByClassName('hm');
    for (var i = 0; i < cells.length; i++) {
        cells[i].addEventListener('mouseenter', function() {
            for (var j = 0; j < cells.length; j++) {
                if (cells[j] !== this) {
                    cells[j].style.opacity = 0.3;
                }
            }
        });
        cells[i].addEventListener('mouseleave', function() {
            for (var j = 0; j < cells.length; j++) {
                cells[j].style.opacity = 1.0;
            }
        });
    }
</script>
"""
st.markdown(hover_code, unsafe_allow_html=True)

# Streamlit app
st.title('Custom Heatmap with Hover Effect in Streamlit')

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)



###################################### Age bullshit #####################################


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
