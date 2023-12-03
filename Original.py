import streamlit as st
import pandas as pd
import plost
import streamlit.components.v1 as components
import os 

# Set the layout of the app
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard')

# Sidebar setup for various parameters
st.sidebar.subheader('Heat map parameter')
time_hist_color = st.sidebar.selectbox('Color by', ('temp_min', 'temp_max')) 

st.sidebar.subheader('Donut chart parameter')
donut_theta = st.sidebar.selectbox('Select data', ('q2', 'q3'))

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['temp_min', 'temp_max'], ['temp_min', 'temp_max'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [Ayush]
''')

# Row A: Metrics Display
st.markdown('### Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

# Row B: Heatmap and Donut Chart
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    plost.time_hist(
        data=seattle_weather,
        date='date',
        x_unit='week',
        y_unit='day',
        color=time_hist_color,
        aggregate='median',
        legend=None,
        height=345,
        use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

# Row C: Line Chart
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)

# Function to load and process noise data
def load_and_process_noise_data(file_paths):
    processed_dfs = {}
    for file_path in file_paths:
        df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
        df = df.groupby('date')['SoundLevel(dB)'].mean().reset_index()
        processed_dfs[file_path] = df
    return processed_dfs

# Load noise data
noise_files = [
    r"BlessingtonNoise.csv",
    r"ChanceryParkNoise.csv",
    r"DrumcondraLibraryNoise.csv",
    r"WoodStockGardensNoise.csv"
]
noise_data = load_and_process_noise_data(noise_files)

# Row D: Noise Level Trend
st.markdown('### Noise Level Trend Analysis')
for file_name, df in noise_data.items():
    location_name = file_name.split('Noise.csv')[0]
    st.markdown(f'#### {location_name}')
    st.line_chart(df, x='date', y='SoundLevel(dB)', height=plot_height)

# Sidebar option to select an HTML file for Heatmap
st.sidebar.subheader('Heatmap Display')
selected_html_file = st.sidebar.selectbox('Select HTML File', os.listdir(r"C:\Users\ayush\Desktop\dashboard-v2-master\Heatmaps"))
if selected_html_file:
    html_file_path = os.path.join(r"C:\Users\ayush\Desktop\dashboard-v2-master\Heatmaps", selected_html_file)
    with open(html_file_path, 'r') as f:
        html_content = f.read()
        components.html(html_content, height=600, scrolling=True)
