# Imports
# Ignoring warnings
import warnings

warnings.filterwarnings('ignore')

# Importing Streamlit
import streamlit as st

# Data Manipulation
import pandas as pd
import numpy as np

from Functions import generate_dates
from Functions import create_features_irr, create_features_win

# Visualization
from PIL import Image
import plotly_express as px

# Model
from xgboost import XGBRegressor
import xgboost

# Automation
import joblib

image = Image.open('Images/energy.jpg')
image1 = Image.open('Images/irradiance.png')

col1, col2 = st.columns([3, 1])

with col1:
    st.title('Energy.AI (Beta Version)')

with col2:
    st.image(image)

st.markdown(""":red[Energy.AI] leverages AI and machine learning to revolutionize solar energy forecasting in Nigeria.
With accurate predictions of solar PV system performance, our app empowers users to harness renewable 
energy effectively. By providing real-time data and advanced algorithms, we offer valuable insights into 
solar irradiation and system efficiency. Tailored to meet the needs of homeowners and commercial entities, 
our comprehensive forecasts optimize energy generation and usage. Join us in unlocking the full potential of 
solar energy in Nigeria for a greener and more sustainable future.""")

col3, col4 = st.columns([3, 1])

with col3:
    st.image(image1, width=500)

with col4:
    st.write('Overview of Solar Irradiance distribution across the Country')

# Variables
tech = ['Crystalline Silicon', 'Copper Indium Gallium Selenide', 'Cadmium Telluride']

states = ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
          "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi",
          "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba",
          "Yobe", "Zamfara", "FCT"]

st.write('How to use app: '
         '\n 1. Select a State.'
         '\n 2. Select a Panel Technology (we produce output for just one Panel typically 1.6764x1.016m for a '
         'residential panel.)'
         '\n 3. Select forecast duration.'
         '\n 4. Visualize results below')

# Selecting Preferred Technology
option_state = st.selectbox('Search or Scroll to Select a State', options=states)
option_PV_tech = st.selectbox('What\'s your preferred technology of Panel ?', options=tech)

col5, col6 = st.columns(2)

with col5:
    start_date = st.date_input('Enter Start Date')

with col6:
    end_date = st.date_input('Enter End Date')

dates_to_forecast = generate_dates(start_date, end_date)

# Writing into a DataFrame
data = pd.DataFrame({'date': dates_to_forecast})

# st.dataframe(dates)
# st.write('here')
#st.write(data)

# Modelling Features
features = ['dayofyear', 'hour', 'dayofweek', 'quater', 'month', 'year']

# Opening Region data based on user input to get train data
state_data_irr = pd.read_csv('data/irradiance/' + option_state + '.csv')
state_data_temp_and_win = pd.read_csv('data/temperature_and_wind_speed/' + option_state + '.csv')
#st.write(state_data_temp_and_win)

# Getting the dependent(y) and Independent(x) Variables
# Irradiance
x_irr = create_features_irr(state_data_irr)
y_irr = state_data_irr['Clear sky GHI']

# Temperature
x_temp = create_features_win(state_data_temp_and_win)
y_temp = state_data_temp_and_win['T2M']

# WindSpeed
x_win = create_features_win(state_data_temp_and_win)
y_win = state_data_temp_and_win['WS10M']

# test data
x_test = create_features_win(data)

# Importing Models for Forecasting Purposes
irrad_mod = joblib.load('models/Irradiance_model.joblib')
temp_mod = joblib.load('models/Temperature_model.joblib')
winsp_mod = joblib.load('models/Windspeed_model.joblib')

# Fitting Models
irrad_mod.fit(x_irr, y_irr)
temp_mod.fit(x_temp, y_temp)
winsp_mod.fit(x_win, y_win)

# Predictions
irrad_pred = irrad_mod.predict(x_test)
temp_pred = temp_mod.predict(x_test)
win_pred = winsp_mod.predict(x_test)

# Adding to the data dataframe
data['Clear Sky GHI'] = irrad_pred
data['Temperature'] = temp_pred
data['Wind_Speed'] = win_pred

#st.dataframe(data)

if option_PV_tech == tech[0]:
    k1, k2, k3, k4, k5, k6 = (-0.017237, -0.040465, -0.004702, 0.000149, 0.000170, 0.000005)
    U0, U1 = (20, 3.2)
elif option_PV_tech == tech[1]:
    k1, k2, k3, k4, k5, k6 = (-0.005554, -0.038724, -0.003723, -0.000905, -0.001256, 0.000001)
    U0, U1 = (20.0, 2.0)
elif option_PV_tech == tech[2]:
    k1, k2, k3, k4, k5, k6 = (-0.046689, -0.072844, -0.002262, 0.000276, 0.000159, -0.000006)
    U0, U1 = (20, 3.2)

# Variables
Area = 1.703
effnom = 20

tm = data['Temperature'] + (data['Clear Sky GHI'] / (U0 + U1 * data['Wind_Speed']))
g = data['Clear Sky GHI']
g_prime = data['Clear Sky GHI'] / 1000
t_prime = tm - 25

effrel = 1 + (k1 * np.log(g_prime)) + (k2 * np.log(g_prime) ** 2) + (k3 * t_prime) + (
        k4 * t_prime * np.log(g_prime)) + (k5 * t_prime * np.log(g_prime) ** 2) + k6 * (t_prime) ** 2

# Calculating Power Output
data['Power_Output'] = g_prime * Area * effnom * effrel

st.write('\n **Below displays Interactive Plots showing Forecasted Solar Irradiance and Calculated Power Output for the'
         ' Specified duration**')
st.subheader('For ' + option_state)
plot = px.line(data, x=data['date'], y=data['Clear Sky GHI'], markers='.', width=500, height=300,
               labels={  # replaces default labels by column name
                   "date": "date", "Clear Sky GHI": "Solar Irradiance (W/m²)"})
st.plotly_chart(plot)

plot = px.line(data, x=data['date'], y=data['Power_Output'], markers='.', width=500, height=300,
               labels={  # replaces default labels by column name
                   "date": "date", "Power_Output": "Power (W)"})
st.plotly_chart(plot)

# Indicators
st.write('**Indicators**')
# Data to show Need Level
need_level = pd.read_csv('data/indicators/need_level.csv')
need_value = need_level[need_level['State'] == option_state]['need level'].values[0]

wealth_ind = pd.read_csv('data/indicators/poverty_level.csv')
wealth_val = wealth_ind[wealth_ind['State'] == option_state]['Pov'].values[0]

col7, col8 = st.columns(2)

with col7:
    st.write('Pecentage of People without Electricity Access')
    st.progress(round(need_value), text=str(round(need_value)) + '%')
    # st.progress(need_level[need_level['State'] == option_state]['need level'].values[0])

with col8:
    st.write('Poverty Level')
    st.progress(round(wealth_val), text=str(round(wealth_val)) + ' million poor people')
    # end_date = st.date_input('Enter End Date')

