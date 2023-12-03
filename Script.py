import pandas as pd
import streamlit as st
import plost
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os 

# Set the layout of the app
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard')

st.title('Noise Levels')
st.markdown("#### Since Yesterday")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Blessington", "67.18 dB ", "-19.5 dB")
col2.metric("ChanceryPark", "60.44 dB", "3 dB")
col3.metric("Drumcondra", "38.71 dB", "-4 dB")
col4.metric("Woodstock", "47.20 dB", "0.34 dB")

# Sidebar option to select an HTML file for Heatmap
st.title("Heatmap Correlating the Dublin Bikes and Noise Sensors")
st.sidebar.subheader('Heatmap Display')
selected_html_file = st.sidebar.selectbox('Select HTML File', os.listdir(r"Heatmaps"))
if selected_html_file:
    html_file_path = os.path.join(r"Heatmaps", selected_html_file)
    with open(html_file_path, 'r') as f:
        html_content = f.read()
        components.html(html_content, height=600, scrolling=True)

# Function to load and process noise data
def load_and_process_noise_data(file_path, location_name):
    try:
        df = pd.read_csv(file_path, parse_dates=['date'], dayfirst=True)
    except ValueError:
        df = pd.read_csv(file_path, parse_dates=['DATE'], dayfirst=True)
        df.rename(columns={'DATE': 'date'}, inplace=True)
    df = df.groupby('date')['SoundLevel(dB)'].mean().reset_index()
    df['Location'] = location_name
    return df

# Function to load combined data (bike usage and environmental factors)
def load_combined_data(file_path):
    return pd.read_csv(file_path)

def load_bike_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['DATE'])
    return df

def main():
    st.title("Combined Noise Level Trend Analysis")

    # Load and process noise data
    noise_files = {
        "BlessingtonNoise.csv": "Blessington",
        "ChanceryParkNoise.csv": "Chancery Park",
        "DrumcondraLibraryNoise.csv": "Drumcondra",
        "WoodStockGardensNoise.csv": "Woodstock"
    }

    noise_data = pd.DataFrame()
    for file, location in noise_files.items():
        df = load_and_process_noise_data(file, location)
        noise_data = pd.concat([noise_data, df])

    # Load combined data (bike and environmental factors)
    combined_data = load_combined_data('Combined_data.csv')  

    # Plotting combined noise data
    plost.line_chart(
        data=noise_data,
        x='date',
        y='SoundLevel(dB)',
        color='Location',
        use_container_width=True
    )

    # Plotting Bike Usage and Noise Level Over Time with plost
    st.markdown('### Bike Usage and Noise Level Over Time')
    image_path = 'Bike Usage and Noise Level Over Time.png'  
    st.image(image_path, use_column_width=True)
    # Plot for Available Bikes
    # plost.line_chart(
    #     data=combined_data,
    #     x='DATE',
    #     y='AVAILABLE_BIKES',
    #     use_container_width=True
    # )
    

    
    bike_data = load_bike_data('Bikes.csv')  

    
    st.markdown('### Time-Based Bike Usage Trends')
    plost.line_chart(
        data=bike_data,
        x='DATE',
        y='AVAILABLE_BIKES',
        use_container_width=True
    )

    
    st.markdown('### Correlation Matrix')
    corr_matrix = combined_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    st.pyplot(plt)

     
    st.markdown('### Geographic Distribution of Bike Usage')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=bike_data, x='LONGITUDE', y='LATITUDE', size='AVAILABLE_BIKES', hue='AVAILABLE_BIKES')
    plt.title('Geographic Distribution of Bike Usage')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    st.pyplot(plt)

    

if __name__ == "__main__":
    main()
