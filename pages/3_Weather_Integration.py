import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import pytz
from weather_api_new import WeatherAPI
from dotenv import load_dotenv
import os

# Configure the page
st.set_page_config(
    page_title="Weather Integration | Solar-Wind Hybrid Monitor",
    page_icon="🌤️",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Timezone setup
tz = pytz.timezone("Africa/Harare")

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "weather_location" not in st.session_state:
    st.session_state.weather_location = os.getenv("LOCATION", "Bulawayo")
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(tz)
if "weather_api" not in st.session_state:
    st.session_state.weather_api = WeatherAPI()

# Custom CSS
st.markdown("""
<style>
    /* Your existing CSS styles */
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Zimbabwe Weather Integration")
st.write("Live Zimbabwe weather data and forecasts to optimize your energy generation")

# Color mapping function
def get_status_color(value, thresholds):
    if value >= thresholds["green"][0]:
        return "#4CAF50"
    elif value >= thresholds["yellow"][0]:
        return "#FFC107"
    else:
        return "#F44336"

# Weather data fetching
def refresh_weather_data():
    location_data = {
        'latitude': -20.15,  # Bulawayo coordinates
        'longitude': 28.58,
        'name': st.session_state.weather_location
    }
    current = st.session_state.weather_api.get_current_weather(location_data)
    forecast = st.session_state.weather_api.get_forecast(location_data)
    return {
        "current": current,
        "forecast": forecast,
        "timestamp": datetime.now(tz)
    }

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Auto-refresh interval (sec)", 5, 60, 15)
    
    if st.button("Refresh Data Now"):
        st.session_state.weather_data = refresh_weather_data()
        st.session_state.last_refresh_time = datetime.now(tz)
    
    available_locations = ["Bulawayo", "Harare", "Victoria Falls"]
    selected_location = st.selectbox(
        "Select Location",
        available_locations,
        index=available_locations.index(st.session_state.weather_location)
    )
    
    if selected_location != st.session_state.weather_location:
        st.session_state.weather_location = selected_location
        st.session_state.weather_data = refresh_weather_data()
        st.rerun()
    
    with st.expander("API Settings"):
        api_key = st.text_input(
            "RapidAPI Key",
            value=os.getenv("RAPIDAPI_KEY", ""),
            type="password"
        )
        if api_key and api_key != os.getenv("RAPIDAPI_KEY"):
            os.environ["RAPIDAPI_KEY"] = api_key
            st.session_state.weather_api.api_key = api_key
            st.session_state.weather_data = refresh_weather_data()

# Auto-refresh logic
elapsed = (datetime.now(tz) - st.session_state.last_refresh_time).total_seconds()
if auto_refresh and elapsed >= refresh_interval:
    st.session_state.weather_data = refresh_weather_data()
    st.session_state.last_refresh_time = datetime.now(tz)

# Initialize data
if "weather_data" not in st.session_state:
    st.session_state.weather_data = refresh_weather_data()

current_weather = st.session_state.weather_data["current"]
forecast = st.session_state.weather_data["forecast"]
last_refresh = st.session_state.last_refresh_time

# Display last refresh time
st.caption(f"Last updated: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Current weather conditions
st.subheader("Current Weather Conditions")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="custom-card solar-card">
        <h3>☀️ Solar Irradiance</h3>
        <h2 style='color: {get_status_color(current_weather['irradiance'], {"green": (600, float('inf')), "yellow": (200, 600), "red": (0, 200)})}'>
            {current_weather['irradiance']:.1f} W/m²
        </h2>
        <p>{current_weather['weather_description']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="custom-card wind-card">
        <h3>💨 Wind Speed</h3>
        <h2 style='color: {get_status_color(current_weather['wind_speed'], {"green": (4, float('inf')), "yellow": (2, 4), "red": (0, 2)})}'>
            {current_weather['wind_speed']:.1f} m/s
        </h2>
        <p>Direction: {current_weather['wind_direction']}°</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="custom-card temp-card">
        <h3>🌡️ Temperature</h3>
        <h2 style='color: {get_status_color(current_weather['temperature'], {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)})}'>
            {current_weather['temperature']:.1f}°C
        </h2>
        <p>Humidity: {current_weather['humidity']}%</p>
    </div>
    """, unsafe_allow_html=True)

# 7-Day Forecast
st.subheader("7-Day Weather Forecast")
st.markdown("---")

forecast_cols = st.columns(7)
for i, day in enumerate(forecast[:7]):
    with forecast_cols[i]:
        st.markdown(f"""
        <div class="forecast-card">
            <h4>{day['day_name']}</h4>
            <p>{day['date']}</p>
            <div style='font-size:1.8rem;'>
                {day['weather_icon']}
            </div>
            <div>
                {day['temperature']:.1f}°C
            </div>
            <div>
                ☀️ {day['irradiance']:.0f} W/m²
            </div>
            <div>
                💨 {day['wind_speed']:.1f} m/s
            </div>
        </div>
        """, unsafe_allow_html=True)

# Historical Data
st.subheader("Historical Weather Data")
st.markdown("---")

hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Temperature & Irradiance", "Wind Data", "Energy Generation"])

with hist_tab1:
    st.line_chart(
        pd.DataFrame(forecast[:7]),
        x="date",
        y=["temperature", "irradiance"],
        color=["#FF5252", "#FFD600"]
    )

with hist_tab2:
    st.line_chart(
        pd.DataFrame(forecast[:7]),
        x="date",
        y=["wind_speed", "cloud_cover"],
        color=["#2962FF", "#9E9E9E"]
    )

with hist_tab3:
    st.line_chart(
        pd.DataFrame(forecast[:7]),
        x="date",
        y=["solar_power", "wind_power"],
        color=["#FFD600", "#2962FF"]
    )

# Weather-based optimization
st.subheader("Weather-Based Energy Optimization")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    st.markdown("### Today's Generation Strategy")
    strategy = "Solar Priority" if current_weather['irradiance'] > 500 else "Wind Priority" if current_weather['wind_speed'] > 4 else "Balanced"
    st.markdown(f"""
    <div style="padding:15px; border-radius:5px;">
        <h3>{strategy}</h3>
    </div>
    """, unsafe_allow_html=True)

with opt_col2:
    st.markdown("### 5-Day Energy Forecast")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[day['date'] for day in forecast[:5]],
        y=[day['solar_potential'] for day in forecast[:5]],
        name="Solar"
    ))
    fig.add_trace(go.Bar(
        x=[day['date'] for day in forecast[:5]],
        y=[day['wind_potential'] for day in forecast[:5]],
        name="Wind"
    ))
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh JavaScript
st.markdown("""
<script>
    setTimeout(function() {
        window.location.reload();
    }, 300000);
</script>
""", unsafe_allow_html=True)