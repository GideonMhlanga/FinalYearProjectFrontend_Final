import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
import time
from utils import get_status_color
from weather_api_new import weather_api  # Our new weather API client
from dotenv import load_dotenv
import pytz
import os
import logging

# Load environment variables
load_dotenv()

# Timezone setup
tz = pytz.timezone("Africa/Harare")

# Configure the page
st.set_page_config(
    page_title="Weather Integration | Solar-Wind Hybrid Monitor",
    page_icon="🌤️",
    layout="wide"
)

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "weather_location" not in st.session_state:
    st.session_state.weather_location = os.getenv("LOCATION", "Bulawayo")
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(tz)

# Title and description
st.title("Zimbabwe Weather Integration")
st.write("Live Zimbabwe weather data and forecasts to optimize your energy generation")

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

# Auto-refresh configuration
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
refresh_interval = st.sidebar.slider("Auto-refresh interval (sec)", 5, 60, 15)

if st.sidebar.button("Refresh Now"):
    st.session_state.weather_data = refresh_weather_data()
    st.session_state.last_refresh_time = datetime.now(tz)

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

# Create columns for current conditions
col1, col2, col3 = st.columns(3)

# Solar irradiance
irradiance = current_weather['irradiance']
irradiance_color = get_status_color(irradiance, {"green": (600, float('inf')), "yellow": (200, 600), "red": (0, 200)})

with col1:
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 5px; background-color: {'#fff9e6' if st.session_state.theme == 'light' else '#332e1f'};">
            <h3 style="margin:0;">☀️ Solar Irradiance</h3>
            <h2 style="margin:0; color: {'green' if irradiance_color == 'green' else 'orange' if irradiance_color == 'yellow' else 'red'};">
                {irradiance:.1f} W/m²
            </h2>
            <p style="margin:0;">
                {{
                    "Excellent for solar generation" if irradiance > 800 else
                    "Good for solar generation" if irradiance > 500 else
                    "Moderate solar potential" if irradiance > 200 else
                    "Low solar potential"
                }}
            </p>
            <p style="margin:0; font-size: 0.8em;">
                {current_weather['conditions']} ({current_weather['description']})
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Wind speed
wind_speed = current_weather['wind_speed']
wind_color = get_status_color(wind_speed, {"green": (4, float('inf')), "yellow": (2, 4), "red": (0, 2)})

with col2:
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 5px; background-color: {'#e6f2ff' if st.session_state.theme == 'light' else '#1a2833'};">
            <h3 style="margin:0;">💨 Wind Speed</h3>
            <h2 style="margin:0; color: {'green' if wind_color == 'green' else 'orange' if wind_color == 'yellow' else 'red'};">
                {wind_speed:.1f} m/s
            </h2>
            <p style="margin:0;">
                {{
                    "Excellent for wind generation" if wind_speed > 6 else
                    "Good for wind generation" if wind_speed > 4 else
                    "Moderate wind potential" if wind_speed > 2 else
                    "Low wind potential"
                }}
            </p>
            <p style="margin:0; font-size: 0.8em;">
                Direction: {current_weather['wind_direction'] or 'N/A'}°
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Temperature
temperature = current_weather['temperature']
temp_color = get_status_color(temperature, {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)})

with col3:
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 5px; background-color: {'#e6ffe6' if st.session_state.theme == 'light' else '#1a331a'};">
            <h3 style="margin:0;">🌡️ Temperature</h3>
            <h2 style="margin:0; color: {'green' if temp_color == 'green' else 'orange' if temp_color == 'yellow' else 'red'};">
                {temperature:.1f} °C
            </h2>
            <p style="margin:0;">
                {{
                    "Optimal for system performance" if 15 <= temperature <= 25 else
                    "Too hot - reduced efficiency" if temperature > 25 else
                    "Too cold - reduced efficiency"
                }}
            </p>
            <p style="margin:0; font-size: 0.8em;">
                Feels like: {current_weather['feels_like']:.1f}°C
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Location selector in sidebar
available_locations = weather_api.get_available_locations()
selected_location = st.sidebar.selectbox(
    "Select Zimbabwe Location",
    available_locations,
    index=available_locations.index(st.session_state.weather_location)
)

if selected_location != st.session_state.weather_location:
    st.session_state.weather_location = selected_location
    st.session_state.weather_data = refresh_weather_data()
    st.rerun()

# API Key Input
with st.sidebar.expander("API Settings"):
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

# Weather forecast section
st.subheader(f"5-Day Weather Forecast for {st.session_state.weather_location}")

# Create forecast cards
forecast_cols = st.columns(len(forecast))

# Weather icon mapping
icon_mapping = {
    "Clear": "☀️",
    "Clouds": "☁️",
    "Rain": "🌧️",
    "Drizzle": "🌦️",
    "Thunderstorm": "⛈️",
    "Snow": "❄️",
    "Mist": "🌫️"
}

# Function to create a weather card
def create_weather_card(day):
    """Create a weather card for a single day."""
    try:
        # Get weather icon based on conditions
        icon = icon_mapping.get(day['conditions'], "🌤️")
        
        # Set background color based on day of week
        day_colors = {
            "Sunday": "#fff9e6" if st.session_state.theme == "light" else "#332e1f",
            "Monday": "#e6f2ff" if st.session_state.theme == "light" else "#1a2833",
            "Tuesday": "#e6ffe6" if st.session_state.theme == "light" else "#1a331a",
            "Wednesday": "#fff0e6" if st.session_state.theme == "light" else "#33261f",
            "Thursday": "#f0e6ff" if st.session_state.theme == "light" else "#261f33",
            "Friday": "#ffe6e6" if st.session_state.theme == "light" else "#331f1f",
            "Saturday": "#e6e6ff" if st.session_state.theme == "light" else "#1f1f33"
        }
        
        bg_color = day_colors.get(day['day_name'], "#f5f5f5" if st.session_state.theme == "light" else "#2d2d2d")
        
        # Format temperature with proper error handling
        temp = day.get('temperature', 0)
        temp_high = day.get('temp_high', temp + 5)
        temp_low = day.get('temp_low', temp - 5)
        
        # Format irradiance with proper error handling
        irradiance = day.get('irradiance', 0)
        irradiance_text = f"{irradiance:.0f} W/m²" if irradiance > 0 else "0 W/m²"
        
        # Create the card
        card = f"""
        <div style="position: relative; padding: 15px; border-radius: 10px; background-color: {bg_color}; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin:0; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{day['day_name']}</h3>
            <p style="margin:0; font-size: 0.8em; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">{day['date']}</p>
            <h1 style="margin:10px 0; font-size: 2.5em;">{icon}</h1>
            <h3 style="margin:5px 0; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{day['conditions']}</h3>
            <p style="margin:5px 0; font-size: 0.9em; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">{day['description']}</p>
            <p style="margin:5px 0; font-size: 1.2em; font-weight: bold; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{temp:.1f}°C</p>
            <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">{temp_low:.1f}°C - {temp_high:.1f}°C</p>
            <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">Solar: {irradiance_text}</p>
            <div style="margin-top: 10px; padding: 8px; border-radius: 5px; background-color: rgba(0,0,0,0.05);">
                <p style="margin:3px 0; font-size: 0.9em;">Wind: {day['wind_speed']:.1f} m/s</p>
                <p style="margin:3px 0; font-size: 0.9em;">Clouds: {day['cloud_cover']:.0f}%</p>
                <p style="margin:3px 0; font-size: 0.9em;">Rain: {day['rain_probability']*100:.0f}%</p>
            </div>
            <div style="margin-top: 8px; padding: 8px; border-radius: 5px; background-color: rgba(0,0,0,0.05);">
                <p style="margin:3px 0; font-size: 0.9em;">Solar Potential: <span style="color: {'green' if day['solar_potential'] == 'High' else 'orange' if day['solar_potential'] == 'Medium' else 'red'}">{day['solar_potential']}</span></p>
                <p style="margin:3px 0; font-size: 0.9em;">Wind Potential: <span style="color: {'green' if day['wind_potential'] == 'High' else 'orange' if day['wind_potential'] == 'Medium' else 'red'}">{day['wind_potential']}</span></p>
            </div>
        </div>
        """
        return card
    except Exception as e:
        logging.error(f"Error creating weather card: {str(e)}")
        return ""

# Check if API is working
is_api_working = True
try:
    test_data = weather_api.get_current_weather()
    if not test_data:
        is_api_working = False
except Exception as e:
    is_api_working = False
    logging.error(f"API test failed: {str(e)}")

# Display API status
if not is_api_working:
    st.error("⚠️ Weather API is currently offline. Displaying simulated data.")

# Create forecast cards
forecast_cols = st.columns(len(forecast))

for i, day in enumerate(forecast):
    with forecast_cols[i]:
        card = create_weather_card(day)
        st.markdown(card, unsafe_allow_html=True)

# Weather impact on energy generation
st.subheader("Weather Impact on Energy Generation")

# Create tabs for different analyses
tab1, tab2 = st.tabs(["Solar vs. Weather", "Wind vs. Weather"])

# Generate mock historical data (replace with real data from your system)
def generate_historical_data(current, forecast):
    """Generate mock historical data based on current conditions and forecast"""
    dates = pd.date_range(end=datetime.now(tz), periods=30, freq="D")
    data = []
    
    for i, date in enumerate(dates):
        # Base values on current conditions with some variation
        temp_variation = np.random.normal(0, 3)
        wind_variation = np.random.normal(0, 1)
        cloud_variation = np.random.normal(0, 10)
        
        temp = current['temperature'] + temp_variation
        wind = current['wind_speed'] + wind_variation
        clouds = current['cloud_cover'] + cloud_variation
        
        # Ensure realistic ranges
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
            "solar_power": max(0, irradiance * 0.15 * (1 + np.random.normal(0, 0.1))),  # 15% efficiency with variation
            "wind_power": max(0, wind**3 * 0.5 * (1 + np.random.normal(0, 0.2)))  # Cubic relationship with variation
        })
    
    return pd.DataFrame(data)

historical_data = generate_historical_data(current_weather, forecast)

# Tab 1: Solar vs. Weather
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
    
    # Add trendline
    try:
        x = historical_data["irradiance"]
        y = historical_data["solar_power"]
        coefficients = np.polyfit(x, y, 1)
        polynomial = np.poly1d(coefficients)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=polynomial(x),
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Trendline"
            )
        )
    except:
        pass
    
    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
        font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA")
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Wind vs. Weather
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
    
    # Add cubic relationship model
    x_range = np.linspace(0, historical_data["wind_speed"].max(), 100)
    y_model = 0.5 * x_range**3  # Simple cubic model
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_model,
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="Power Curve Model"
        )
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
        font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA")
    )
    st.plotly_chart(fig, use_container_width=True)

# Weather-based energy optimization
st.subheader("Weather-Based Energy Optimization")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    st.markdown("### Today's Generation Strategy")
    
    # Determine optimal strategy
    if irradiance > 500 and wind_speed < 3:
        strategy = "Solar Priority"
        explanation = "High solar irradiance and low wind speed indicate solar will be your primary generation source today."
        icon = "☀️"
        color = "#FFD700"
    elif irradiance < 300 and wind_speed > 4:
        strategy = "Wind Priority"
        explanation = "Low solar irradiance and good wind speed indicate wind will be your primary generation source today."
        icon = "💨"
        color = "#4682B4"
    else:
        strategy = "Balanced"
        explanation = "Current conditions support both solar and wind generation. A balanced approach is recommended."
        icon = "⚖️"
        color = "#9370DB"
    
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 5px; border: 2px solid {color}; background-color: {'rgba(255,255,255,0.1)' if st.session_state.theme == 'dark' else 'rgba(0,0,0,0.05)'};">
            <h2 style="margin:0; color: {color};">{icon} {strategy}</h2>
            <p style="margin-top: 10px;">{explanation}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Recommendations
    st.markdown("### Recommendations")
    if strategy == "Solar Priority":
        recs = [
            "Ensure solar panels are clean and unobstructed",
            "Maximize solar exposure by adjusting panel angles if possible",
            "Schedule high-energy activities during peak sun hours",
            "Store excess energy in batteries for night use"
        ]
    elif strategy == "Wind Priority":
        recs = [
            "Verify wind turbine maintenance is up to date",
            "Clear any potential obstructions around turbines",
            "Schedule high-energy activities during peak wind periods",
            "Store excess energy in batteries for low-wind periods"
        ]
    else:
        recs = [
            "Balance loads between both energy sources",
            "Monitor both systems for optimal performance",
            "Dynamically adjust consumption based on generation",
            "Maintain batteries at moderate charge level"
        ]
    
    for rec in recs:
        st.markdown(f"- {rec}")

with opt_col2:
    st.markdown("### 5-Day Energy Forecast")
    
    # Prepare data for chart
    dates = [day['date'] for day in forecast]
    solar_potential = [100 if day['solar_potential'] == "High" else 60 if day['solar_potential'] == "Medium" else 30 for day in forecast]
    wind_potential = [100 if day['wind_potential'] == "High" else 60 if day['wind_potential'] == "Medium" else 30 for day in forecast]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=solar_potential,
        name="Solar Potential",
        marker_color="#FFD700"
    ))
    fig.add_trace(go.Bar(
        x=dates,
        y=wind_potential,
        name="Wind Potential",
        marker_color="#4682B4"
    ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Energy Potential (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=30, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
        font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA"),
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekly planning recommendations
    best_solar_day = dates[solar_potential.index(max(solar_potential))]
    best_wind_day = dates[wind_potential.index(max(wind_potential))]
    worst_day = dates[(np.array(solar_potential) + np.array(wind_potential)).argmin()]
    
    st.markdown(f"""
    - **Best day for solar generation:** {best_solar_day}
    - **Best day for wind generation:** {best_wind_day}
    - **Energy conservation day:** {worst_day}
    """)
    
    with st.expander("Energy Conservation Plan for Low Generation Days"):
        st.markdown("""
        1. Shift non-essential loads to high-generation days
        2. Reduce discretionary energy use (HVAC settings, etc.)
        3. Pre-charge battery to maximum capacity the day before
        4. Prioritize critical loads with automated load shedding
        5. Activate backup systems if available
        """)

# Weather alert system
with st.expander("Weather Alerts and Notifications", expanded=False):
    st.markdown("### Configure Weather Alerts")
    
    # Alert settings
    alert_cols = st.columns(3)
    with alert_cols[0]:
        st.checkbox("High wind alerts (>15 m/s)", value=True)
        st.checkbox("Low irradiance alerts (<100 W/m²)", value=True)
    with alert_cols[1]:
        st.checkbox("Extreme temperature alerts", value=True)
        st.checkbox("Heavy precipitation alerts", value=True)
    with alert_cols[2]:
        st.checkbox("Favorable generation alerts", value=False)
        st.checkbox("Daily forecast summary", value=True)
    
    # Notification methods
    st.markdown("### Notification Methods")
    notify_cols = st.columns(3)
    with notify_cols[0]:
        st.checkbox("Dashboard alerts", value=True)
    with notify_cols[1]:
        email = st.checkbox("Email alerts", value=False)
        if email:
            st.text_input("Email address")
    with notify_cols[2]:
        sms = st.checkbox("SMS alerts", value=False)
        if sms:
            st.text_input("Phone number")
    
    st.button("Save Alert Settings")

# Add auto-refresh functionality
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now(pytz.timezone('Africa/Harare'))

# Function to check if we should refresh the data
def should_refresh():
    elapsed = (datetime.now(pytz.timezone('Africa/Harare')) - st.session_state.last_refresh).total_seconds()
    return elapsed >= 300  # Refresh every 5 minutes

# Get current weather data
current_weather = weather_api.get_current_weather()

# Create three columns for current conditions
col1, col2, col3 = st.columns(3)

# Current Weather Card
with col1:
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {'#fff9e6' if st.session_state.theme == 'light' else '#332e1f'}; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin:0; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">Current Weather</h3>
        <h1 style="margin:10px 0; font-size: 2.5em;">{icon_mapping.get(current_weather['conditions'], '🌤️')}</h1>
        <p style="margin:5px 0; font-size: 1.2em; font-weight: bold; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{current_weather['temperature']:.1f}°C</p>
        <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">{current_weather['conditions']}</p>
        <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">{current_weather['description']}</p>
        <div style="margin-top: 10px; padding: 8px; border-radius: 5px; background-color: rgba(0,0,0,0.05);">
            <p style="margin:3px 0; font-size: 0.9em;">Wind: {current_weather['wind_speed']:.1f} m/s</p>
            <p style="margin:3px 0; font-size: 0.9em;">Clouds: {current_weather['cloud_cover']:.0f}%</p>
            <p style="margin:3px 0; font-size: 0.9em;">Humidity: {current_weather['humidity']:.0f}%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Solar Irradiance Card
with col2:
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {'#fff9e6' if st.session_state.theme == 'light' else '#332e1f'}; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin:0; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">Solar Irradiance</h3>
        <h1 style="margin:10px 0; font-size: 2.5em;">☀️</h1>
        <p style="margin:5px 0; font-size: 1.2em; font-weight: bold; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{current_weather['irradiance']:.0f} W/m²</p>
        <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">Solar Potential: {current_weather['solar_potential']}</p>
        <div style="margin-top: 10px; padding: 8px; border-radius: 5px; background-color: rgba(0,0,0,0.05);">
            <p style="margin:3px 0; font-size: 0.9em;">Cloud Cover: {current_weather['cloud_cover']:.0f}%</p>
            <p style="margin:3px 0; font-size: 0.9em;">Time: {current_weather['timestamp'].strftime('%H:%M')}</p>
            <p style="margin:3px 0; font-size: 0.9em;">Last Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Wind Conditions Card
with col3:
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; background-color: {'#fff9e6' if st.session_state.theme == 'light' else '#332e1f'}; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3 style="margin:0; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">Wind Conditions</h3>
        <h1 style="margin:10px 0; font-size: 2.5em;">💨</h1>
        <p style="margin:5px 0; font-size: 1.2em; font-weight: bold; color: {'#333' if st.session_state.theme == 'light' else '#fff'};">{current_weather['wind_speed']:.1f} m/s</p>
        <p style="margin:5px 0; color: {'#666' if st.session_state.theme == 'light' else '#ccc'};">Direction: {current_weather['wind_direction']:.0f}°</p>
        <div style="margin-top: 10px; padding: 8px; border-radius: 5px; background-color: rgba(0,0,0,0.05);">
            <p style="margin:3px 0; font-size: 0.9em;">Wind Potential: {current_weather['wind_potential']}</p>
            <p style="margin:3px 0; font-size: 0.9em;">Pressure: {current_weather['pressure']:.0f} hPa</p>
            <p style="margin:3px 0; font-size: 0.9em;">Last Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add auto-refresh button
if st.button("Refresh Now", key="refresh_weather_button"):
    st.session_state.last_refresh = datetime.now(pytz.timezone('Africa/Harare'))
    st.experimental_rerun()

# Add auto-refresh using JavaScript
st.markdown("""
<script>
    // Auto-refresh the page every 5 minutes
    setTimeout(function() {
        window.location.reload();
    }, 300000);  // 300000 ms = 5 minutes
</script>
""", unsafe_allow_html=True)