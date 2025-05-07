import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Weather Integration | Solar-Wind Hybrid Monitor",
    page_icon="🌤️",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import pytz
from weather_api_new import weather_api
from dotenv import load_dotenv
import os

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

# Custom CSS for the entire app
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Card styling */
    .custom-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    
    /* Current weather cards */
    .solar-card {
        background-color: #FFF9C4;
        border-left: 5px solid #FFC107;
    }
    
    .wind-card {
        background-color: #B3E5FC;
        border-left: 5px solid #2196F3;
    }
    
    .temp-card {
        background-color: #FFCCBC;
        border-left: 5px solid #FF5722;
    }
    
    /* Forecast cards */
    .forecast-card {
        padding: 12px;
        border-radius: 10px;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .expander-content {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    
    .metric-box {
        flex: 1;
        padding: 0 5px;
    }
    
    .metric-title {
        font-weight: bold;
        font-size: 0.8rem;
        color: #555;
    }
    
    .metric-value {
        font-size: 1rem;
        font-weight: bold;
        color: #222;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Zimbabwe Weather Integration")
st.write("Live Zimbabwe weather data and forecasts to optimize your energy generation")

# Color mapping functions
def get_status_color(value, thresholds):
    if value >= thresholds["green"][0]:
        return "#4CAF50"  # Green
    elif value >= thresholds["yellow"][0]:
        return "#FFC107"  # Amber
    else:
        return "#F44336"  # Red

# Function to refresh weather data
def refresh_weather_data():
    try:
        current = weather_api.get_current_weather(st.session_state.weather_location)
        forecast = weather_api.get_forecast(location=st.session_state.weather_location)
        return {
            "current": current,
            "forecast": forecast,
            "timestamp": datetime.now(tz)
        }
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    # Auto-refresh configuration
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Auto-refresh interval (sec)", 5, 60, 15)
    
    if st.button("Refresh Data Now", key="sidebar_refresh_button"):
        st.session_state.weather_data = refresh_weather_data()
        st.session_state.last_refresh_time = datetime.now(tz)
    
    # Location selector
    available_locations = weather_api.get_available_locations()
    selected_location = st.selectbox(
        "Select Location",
        available_locations,
        index=available_locations.index(st.session_state.weather_location)
    )
    
    if selected_location != st.session_state.weather_location:
        st.session_state.weather_location = selected_location
        st.session_state.weather_data = refresh_weather_data()
        st.rerun()
    
    # API Key Input
    with st.expander("API Settings"):
        api_key = st.text_input(
            "RapidAPI Key",
            value=os.getenv("RAPIDAPI_KEY", ""),
            type="password",
            help="Get your API key from RapidAPI marketplace"
        )
        
        if api_key and api_key != os.getenv("RAPIDAPI_KEY"):
            os.environ["RAPIDAPI_KEY"] = api_key
            weather_api.api_key = api_key
            st.success("API key updated successfully!")
            st.session_state.weather_data = refresh_weather_data()

# Check if we need to refresh
elapsed = (datetime.now(tz) - st.session_state.last_refresh_time).total_seconds()
if auto_refresh and elapsed >= refresh_interval:
    st.session_state.weather_data = refresh_weather_data()
    st.session_state.last_refresh_time = datetime.now(tz)

# Get current data or initialize
if "weather_data" not in st.session_state:
    st.session_state.weather_data = refresh_weather_data()

if st.session_state.weather_data is None:
    st.error("Failed to load weather data. Please check your API key and connection.")
    st.stop()

current_weather = st.session_state.weather_data["current"]
forecast = st.session_state.weather_data["forecast"]
last_refresh = st.session_state.last_refresh_time

# Display last refresh time
st.caption(f"Last updated: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Current weather conditions
st.subheader("Current Weather Conditions")
col1, col2, col3 = st.columns(3)

# Solar Card
with col1:
    irradiance_color = get_status_color(current_weather['irradiance'], 
                                      {"green": (600, float('inf')), 
                                       "yellow": (200, 600), 
                                       "red": (0, 200)})
    
    st.markdown(f"""
    <div class="custom-card solar-card">
        <h3>☀️ Solar Irradiance</h3>
        <h2 style='color: {irradiance_color}'>{current_weather['irradiance']:.1f} W/m²</h2>
        <p>{'Excellent' if current_weather['irradiance'] > 800 else 
            'Good' if current_weather['irradiance'] > 500 else 
            'Moderate' if current_weather['irradiance'] > 200 else 'Low'} solar potential</p>
        <p style='color: #666; font-size: 0.9rem;'>{current_weather['conditions']} ({current_weather['description']})</p>
    </div>
    """, unsafe_allow_html=True)

# Wind Card
with col2:
    wind_color = get_status_color(current_weather['wind_speed'],
                                {"green": (4, float('inf')),
                                 "yellow": (2, 4),
                                 "red": (0, 2)})
    
    st.markdown(f"""
    <div class="custom-card wind-card">
        <h3>💨 Wind Speed</h3>
        <h2 style='color: {wind_color}'>{current_weather['wind_speed']:.1f} m/s</h2>
        <p>{'Excellent' if current_weather['wind_speed'] > 6 else 
            'Good' if current_weather['wind_speed'] > 4 else 
            'Moderate' if current_weather['wind_speed'] > 2 else 'Low'} wind potential</p>
        <p style='color: #666; font-size: 0.9rem;'>Direction: {current_weather['wind_direction'] or 'N/A'}°</p>
    </div>
    """, unsafe_allow_html=True)

# Temperature Card
with col3:
    temp_color = get_status_color(current_weather['temperature'],
                                {"green": (15, 25),
                                 "yellow": (5, 15),
                                 "red": (25, 45)})
    
    st.markdown(f"""
    <div class="custom-card temp-card">
        <h3>🌡️ Temperature</h3>
        <h2 style='color: {temp_color}'>{current_weather['temperature']:.1f}°C</h2>
        <p>{'Optimal' if 15 <= current_weather['temperature'] <= 25 else 
            'Too hot' if current_weather['temperature'] > 25 else 'Too cold'} for system performance</p>
        <p style='color: #666; font-size: 0.9rem;'>Feels like: {current_weather['feels_like']:.1f}°C</p>
    </div>
    """, unsafe_allow_html=True)

# 7-Day Forecast with Distinct Colors
st.subheader("7-Day Weather Forecast")
st.markdown("---")

icon_mapping = {
    "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", 
    "Drizzle": "🌦️", "Thunderstorm": "⛈️", 
    "Snow": "❄️", "Mist": "🌫️"
}

# Different pastel colors for each day
forecast_colors = [
    "#FFCDD2", "#F8BBD0", "#E1BEE7", "#D1C4E9", 
    "#C5CAE9", "#BBDEFB", "#B3E5FC"
]

forecast_cols = st.columns(7)
for i, day in enumerate(forecast[:7]):
    with forecast_cols[i]:
        temp_color = get_status_color(day['temperature'],
                                    {"green": (15, 25),
                                     "yellow": (5, 15),
                                     "red": (25, 45)})
        
        st.markdown(f"""
        <div class="forecast-card" style="background-color: {forecast_colors[i]};">
            <h4 style='margin:0;'>{day['day_name']}</h4>
            <p style='color:#666; margin:5px 0; font-size:0.8rem;'>{day['date']}</p>
            <div style='font-size:1.8rem; text-align:center; margin:5px 0;'>
                {icon_mapping.get(day['conditions'], "🌤️")}
            </div>
            <div style='font-weight:bold; text-align:center; color:{temp_color};'>
                {day['temperature']:.1f}°C
            </div>
            <p style='text-align:center; color:#666; font-size:0.7rem; margin:5px 0;'>
                H: {day['temp_high']:.1f}°C<br>L: {day['temp_low']:.1f}°C
            </p>
            <div style='display:flex; justify-content:space-between; margin:5px 0;'>
                <span>☀️</span>
                <span style='font-weight:bold;'>{day['irradiance']:.0f} W/m²</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin:5px 0;'>
                <span>💨</span>
                <span style='font-weight:bold;'>{day['wind_speed']:.1f} m/s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Horizontal expandable section
        with st.expander("Details", expanded=False):
            st.markdown(f"""
            <div class="expander-content">
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-title">Cloud Cover</div>
                        <div class="metric-value">{day['cloud_cover']:.0f}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Rain Chance</div>
                        <div class="metric-value">{day['rain_probability']*100:.0f}%</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="metric-title">Solar Potential</div>
                        <div class="metric-value" style="color: {'#4CAF50' if day['solar_potential'] == 'High' else '#FFC107' if day['solar_potential'] == 'Medium' else '#F44336'}">
                            {day['solar_potential']}
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-title">Wind Potential</div>
                        <div class="metric-value" style="color: {'#4CAF50' if day['wind_potential'] == 'High' else '#FFC107' if day['wind_potential'] == 'Medium' else '#F44336'}">
                            {day['wind_potential']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Historical Data Visualization
st.subheader("Historical Weather Data")
st.markdown("---")

def generate_historical_data(current, forecast):
    dates = pd.date_range(end=datetime.now(tz), periods=30, freq="D")
    data = []
    
    for i, date in enumerate(dates):
        temp_variation = np.random.normal(0, 3)
        wind_variation = np.random.normal(0, 1)
        cloud_variation = np.random.normal(0, 10)
        
        temp = current['temperature'] + temp_variation
        wind = current['wind_speed'] + wind_variation
        clouds = current['cloud_cover'] + cloud_variation
        
        temp = max(min(temp, 40), -5)
        wind = max(min(wind, 20), 0)
        clouds = max(min(clouds, 100), 0)
        
        is_day = current['sunrise'].time() <= date.time() <= current['sunset'].time()
        irradiance = weather_api._calculate_irradiance(clouds, is_day) if is_day else 0
        
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": temp,
            "wind_speed": wind,
            "cloud_cover": clouds,
            "irradiance": irradiance,
            "solar_power": max(0, irradiance * 0.15 * (1 + np.random.normal(0, 0.1))),
            "wind_power": max(0, wind**3 * 0.5 * (1 + np.random.normal(0, 0.2)))
        })
    
    return pd.DataFrame(data)

historical_data = generate_historical_data(current_weather, forecast)
if historical_data is not None and len(historical_data) > 0:
    chart_data = pd.DataFrame(historical_data)
    
    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Temperature & Irradiance", "Wind Data", "Energy Generation"])
    
    with hist_tab1:
        st.line_chart(
            chart_data,
            x="date",
            y=["temperature", "irradiance"],
            color=["#FF5252", "#FFD600"]  # Red and Gold
        )
    
    with hist_tab2:
        st.line_chart(
            chart_data,
            x="date",
            y=["wind_speed", "cloud_cover"],
            color=["#2962FF", "#9E9E9E"]  # Blue and Gray
        )
    
    with hist_tab3:
        st.line_chart(
            chart_data,
            x="date",
            y=["solar_power", "wind_power"],
            color=["#FFD600", "#2962FF"]  # Gold and Blue
        )

# Weather impact on energy generation - with increased padding
st.subheader("Weather Impact on Energy Generation")

tab1, tab2 = st.tabs(["Solar vs. Weather", "Wind vs. Weather"])

with tab1:
    fig = px.scatter(
        historical_data,
        x="irradiance",
        y="solar_power",
        color="temperature",
        color_continuous_scale="Viridis",
        labels={
            "irradiance": "Solar Irradiance (W/m²)",
            "solar_power": "Solar Power (kW)", 
            "temperature": "Temperature (°C)"
        },
        title="Solar Power vs. Irradiance"
    )
    fig.update_layout(
        margin=dict(l=80, r=80, t=80, b=80),  # Increased padding
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(
        historical_data,
        x="wind_speed",
        y="wind_power",
        color="temperature",
        color_continuous_scale="Viridis",
        labels={
            "wind_speed": "Wind Speed (m/s)",
            "wind_power": "Wind Power (kW)", 
            "temperature": "Temperature (°C)"
        },
        title="Wind Power vs. Wind Speed"
    )
    fig.update_layout(
        margin=dict(l=80, r=80, t=80, b=80),  # Increased padding
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Weather-based energy optimization
st.subheader("Weather-Based Energy Optimization")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    st.markdown("### Today's Generation Strategy")
    
    if current_weather['irradiance'] > 500 and current_weather['wind_speed'] < 3:
        strategy = "Solar Priority"
        color = "#FFD600"  # Gold
    elif current_weather['irradiance'] < 300 and current_weather['wind_speed'] > 4:
        strategy = "Wind Priority"
        color = "#2962FF"  # Blue
    else:
        strategy = "Balanced"
        color = "#9C27B0"  # Purple
    
    st.markdown(f"""
    <div style="padding:15px; border-radius:5px; border-left:5px solid {color}; background-color:#f8f9fa;">
        <h3 style="margin:0; color:{color};">{strategy}</h3>
        <p style="margin-top:8px;">{'Focus on solar generation' if strategy == 'Solar Priority' else 
                                  'Focus on wind generation' if strategy == 'Wind Priority' else 
                                  'Balance between solar and wind'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Recommendations")
    if strategy == "Solar Priority":
        st.markdown("""
        - Clean solar panels
        - Optimize panel angles
        - Schedule energy-intensive tasks during peak sun
        """)
    elif strategy == "Wind Priority":
        st.markdown("""
        - Ensure turbines are unobstructed
        - Schedule energy-intensive tasks during windy periods
        - Check turbine maintenance
        """)
    else:
        st.markdown("""
        - Monitor both systems
        - Balance energy storage
        - Be prepared to shift focus
        """)

with opt_col2:
    st.markdown("### 5-Day Energy Forecast")
    
    dates = [day['date'] for day in forecast[:5]]
    solar_potential = [100 if day['solar_potential'] == "High" else 60 if day['solar_potential'] == "Medium" else 30 for day in forecast[:5]]
    wind_potential = [100 if day['wind_potential'] == "High" else 60 if day['wind_potential'] == "Medium" else 30 for day in forecast[:5]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=solar_potential,
        name="Solar",
        marker_color="#FFD600"  # Gold
    ))
    fig.add_trace(go.Bar(
        x=dates,
        y=wind_potential,
        name="Wind",
        marker_color="#2962FF"  # Blue
    ))
    fig.update_layout(
        margin=dict(l=80, r=80, t=80, b=80),  # Increased padding
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    - **Best solar day:** {dates[solar_potential.index(max(solar_potential))]}
    - **Best wind day:** {dates[wind_potential.index(max(wind_potential))]}
    - **Conserve energy on:** {dates[(np.array(solar_potential) + np.array(wind_potential)).argmin()]}
    """)

# Weather alert system
with st.expander("Weather Alerts and Notifications", expanded=False):
    st.markdown("### Configure Weather Alerts")
    alert_cols = st.columns(3)
    with alert_cols[0]:
        st.checkbox("High wind alerts", True)
        st.checkbox("Low irradiance alerts", True)
    with alert_cols[1]:
        st.checkbox("Temperature alerts", True)
        st.checkbox("Precipitation alerts", True)
    with alert_cols[2]:
        st.checkbox("Favorable conditions", False)
        st.checkbox("Daily summary", True)
    
    st.markdown("### Notification Methods")
    notify_cols = st.columns(3)
    with notify_cols[0]:
        st.checkbox("Dashboard", True)
    with notify_cols[1]:
        email = st.checkbox("Email", False)
        if email:
            st.text_input("Email address")
    with notify_cols[2]:
        sms = st.checkbox("SMS", False)
        if sms:
            st.text_input("Phone number")
    
    st.button("Save Settings")

# Auto-refresh JavaScript
st.markdown("""
<script>
    setTimeout(function() {
        window.location.reload();
    }, 300000);  // 5 minutes
</script>
""", unsafe_allow_html=True)