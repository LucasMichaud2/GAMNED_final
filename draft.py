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
