import streamlit as st

# --- Configuration ---
st.set_page_config(
    page_title="Weather Integration | Solar-Wind Hybrid Monitor",
    page_icon="🌤️",
    layout="wide"
)

import sys
import os
from pathlib import Path

# Add the project root directory to Python's module search path
project_root = str(Path(__file__).parent.parent)  # Goes up from /pages to /FinalYearProjectFrontend_Final
sys.path.append(project_root)

# Now you can import weather_api_new from the root directory
from weather_api_new import weather_api

from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from dotenv import load_dotenv
import os

load_dotenv()
tz = pytz.timezone("Africa/Harare")

# --- Smart Cache Implementation ---
class WeatherCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
            cls._instance._timestamps = {}
        return cls._instance
    
    def get_cache_key(self, location, call_type):
        return f"{hashlib.md5(location.encode()).hexdigest()}_{call_type}"
    
    def get_data(self, location, call_type, max_age_minutes=60):
        cache_key = self.get_cache_key(location, call_type)
        
        # Return cached data if available and fresh
        if cache_key in self._cache:
            last_update = self._timestamps.get(cache_key, datetime.min.replace(tzinfo=tz))
            if (datetime.now(tz) - last_update) < timedelta(minutes=max_age_minutes):
                return self._cache[cache_key]
        
        # Fetch new data
        try:
            if call_type == "current":
                data = weather_api.get_current_weather(location)
            elif call_type == "forecast":
                data = weather_api.get_forecast(location)
            else:
                raise ValueError("Invalid call type")
            
            self._cache[cache_key] = data
            self._timestamps[cache_key] = datetime.now(tz)
            return data
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return self._cache.get(cache_key)  # Fallback to cached data if available

weather_cache = WeatherCache()

# --- Session State Initialization ---
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(tz)
if "weather_location" not in st.session_state:
    st.session_state.weather_location = os.getenv("LOCATION", "Bulawayo")
if "api_call_count" not in st.session_state:
    st.session_state.api_call_count = 0
if "last_location" not in st.session_state:
    st.session_state.last_location = None

# --- Helper Functions ---
def get_status_color(value, thresholds):
    """Safely determine status color with type checking"""
    try:
        # Convert value to float if it's a string
        if isinstance(value, str):
            value = float(value)
        
        # Handle None values
        if value is None:
            return "#9E9E9E"  # Grey for missing data
            
        if value >= thresholds["green"][0]:
            return "#4CAF50"
        elif value >= thresholds["yellow"][0]:
            return "#FFC107"
        else:
            return "#F44336"
    except (ValueError, TypeError):
        return "#9E9E9E"  # Grey if conversion fails

def refresh_weather_data(force=False):
    location = st.session_state.weather_location
    
    # Skip refresh if same location and data is fresh
    if not force and st.session_state.last_location == location:
        if (datetime.now(tz) - st.session_state.last_refresh_time) < timedelta(minutes=60):
            return st.session_state.weather_data
    
    try:
        current = weather_cache.get_data(location, "current")
        forecast = weather_cache.get_data(location, "forecast")
        
        # Validate forecast data structure
        if not isinstance(forecast, list):
            forecast = weather_api.get_forecast(location=location)  # Regenerate if invalid
        
        # Only count as API call if cache was stale
        if current != weather_cache._cache.get(weather_cache.get_cache_key(location, "current")):
            st.session_state.api_call_count += 1
        if forecast != weather_cache._cache.get(weather_cache.get_cache_key(location, "forecast")):
            st.session_state.api_call_count += 1
            
        st.session_state.last_location = location
        st.session_state.last_refresh_time = datetime.now(tz)
        
        return {
            "current": current,
            "forecast": forecast,
            "timestamp": datetime.now(tz)
        }
    except Exception as e:
        st.error(f"Refresh failed: {str(e)}")
        return None

# --- Custom CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .custom-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
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
    .forecast-card {
        padding: 12px;
        border-radius: 10px;
        margin: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
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

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Settings")
    
    # API Usage Monitoring
    st.metric("API Calls This Session", st.session_state.api_call_count)
    st.progress(min(st.session_state.api_call_count / 50, 1.0))
    
    # Refresh Controls
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    if st.button("Force Refresh Now"):
        st.session_state.weather_data = refresh_weather_data(force=True)
        st.rerun()
    
    # Location Selector
    available_locations = weather_api.get_available_locations()
    selected_location = st.selectbox(
        "Select Location",
        available_locations,
        index=available_locations.index(st.session_state.weather_location)
    )
    
    if selected_location != st.session_state.weather_location:
        st.session_state.weather_location = selected_location
        st.session_state.weather_data = refresh_weather_data(force=True)
        st.rerun()
    
    # Cache Status
    st.info(f"""
    **Cache Status**  
    - Current: {st.session_state.weather_location}  
    - Last refresh: {st.session_state.last_refresh_time.strftime('%Y-%m-%d %H:%M')}  
    - Next refresh: {(st.session_state.last_refresh_time + timedelta(minutes=60)).strftime('%H:%M')}
    """)

# --- Auto-Refresh Logic ---
if auto_refresh and (datetime.now(tz) - st.session_state.last_refresh_time) > timedelta(minutes=60):
    st.session_state.weather_data = refresh_weather_data()
    st.rerun()

# --- Data Loading ---
if st.session_state.weather_data is None:
    st.session_state.weather_data = refresh_weather_data()

if not st.session_state.weather_data:
    st.error("Failed to load weather data")
    st.stop()

current_weather = st.session_state.weather_data.get("current", {})
forecast = st.session_state.weather_data.get("forecast", [])
last_refresh = st.session_state.last_refresh_time

# --- Main UI ---
st.title("Zimbabwe Weather Integration")
st.write("Live weather data to optimize your hybrid energy generation")

# Last Updated Display with Simulated Refresh
elapsed_min = (datetime.now(tz) - last_refresh).total_seconds() / 60
refresh_in = max(0, 60 - int(elapsed_min))
js_last_refresh = last_refresh.strftime('%Y-%m-%d %H:%M:%S')

st.markdown(f"""
<p class="last-updated" style="color: #666; font-size: 0.9rem;">
    Last updated: {js_last_refresh} 
    (Refreshing in {refresh_in} minutes)
</p>
<script>
    // Initialize with current values
    let lastRefreshTime = "{js_last_refresh}";
    let minutesLeft = {refresh_in};
    
    // Update countdown every minute
    setInterval(() => {{
        minutesLeft = Math.max(0, minutesLeft - 1);
        document.querySelector('.last-updated').textContent = 
            `Last updated: ${{lastRefreshTime}} (Refreshing in ${{minutesLeft}} minutes)`;
        
        // Force refresh when countdown reaches 0 (handled by Streamlit's auto-refresh)
        if (minutesLeft <= 0) {{
            window.location.reload();
        }}
    }}, 60000);
</script>
""", unsafe_allow_html=True)

# Current Weather Cards
st.subheader("Current Weather Conditions")
col1, col2, col3 = st.columns(3)

# Safely extract current weather data with fallback defaults
irradiance = current_weather.get('irradiance')
conditions = current_weather.get('conditions', 'N/A')
description = current_weather.get('description', '')
wind_speed = current_weather.get('wind_speed')
wind_direction = current_weather.get('wind_direction')
temperature = current_weather.get('temperature')
feels_like = current_weather.get('feels_like')
cloud_cover = current_weather.get('cloud_cover')

# Solar Card
with col1:
    if irradiance is not None:
        irradiance_color = get_status_color(
            irradiance, 
            {"green": (600, float('inf')), "yellow": (200, 600), "red": (0, 200)}
        )
        st.markdown(f"""
        <div class="custom-card solar-card">
            <h3>☀️ Solar Irradiance</h3>
            <h2 style='color: {irradiance_color}'>{irradiance:.1f} W/m²</h2>
            <p>{'Excellent' if irradiance > 800 else 'Good' if irradiance > 500 else 'Moderate' if irradiance > 200 else 'Low'} solar potential</p>
            <p style='color: #666; font-size: 0.9rem;'>{conditions} ({description})</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No solar irradiance data available.")

# Wind Card
with col2:
    if wind_speed is not None:
        wind_color = get_status_color(
            wind_speed,
            {"green": (4, float('inf')), "yellow": (2, 4), "red": (0, 2)}
        )
        wind_direction_display = wind_direction if wind_direction is not None else "N/A"
        st.markdown(f"""
        <div class="custom-card wind-card">
            <h3>💨 Wind Speed</h3>
            <h2 style='color: {wind_color}'>{wind_speed:.1f} m/s</h2>
            <p>{'Excellent' if wind_speed > 6 else 'Good' if wind_speed > 4 else 'Moderate' if wind_speed > 2 else 'Low'} wind potential</p>
            <p style='color: #666; font-size: 0.9rem;'>Direction: {wind_direction_display}°</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No wind speed data available.")

# Temperature Card
with col3:
    if temperature is not None:
        temp_color = get_status_color(
            temperature,
            {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)}
        )
        feels_like_display = f"{feels_like:.1f}°C" if feels_like is not None else "N/A"
        temp_description = ('Optimal' if 15 <= temperature <= 25 else
                            'Too hot' if temperature > 25 else
                            'Too cold')
        st.markdown(f"""
        <div class="custom-card temp-card">
            <h3>🌡️ Temperature</h3>
            <h2 style='color: {temp_color}'>{temperature:.1f}°C</h2>
            <p>{temp_description} for system performance</p>
            <p style='color: #666; font-size: 0.9rem;'>Feels like: {feels_like_display}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No temperature data available.")

# 7-Day Forecast
st.subheader("7-Day Weather Forecast")
st.markdown("---")

# Ensure forecast data exists and is in correct format
if not forecast or not isinstance(forecast, list):
    st.warning("No forecast data available")
else:
    # Define icon mapping
    icon_mapping = {
        "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", 
        "Drizzle": "🌦️", "Thunderstorm": "⛈️", 
        "Snow": "❄️", "Mist": "🌫️", "Overcast": "☁️",
        "Partly Cloudy": "⛅"
    }

    # Create columns for each day
    forecast_cols = st.columns(7)
    
    for i, day in enumerate(forecast[:7]):  # Show up to 7 days
        with forecast_cols[i]:
            # Safely get all values with defaults
            day_name = day.get('day_name', 'Day')
            date = day.get('date', '')
            conditions = day.get('conditions', 'Clear')
            temp = float(day.get('temperature', 20))
            temp_high = float(day.get('temp_high', temp + 5))
            temp_low = float(day.get('temp_low', temp - 5))
            wind_speed = float(day.get('wind_speed', 0))
            irradiance = float(day.get('irradiance', 0))
            cloud_cover = day.get('cloud_cover', 0)
            
            # Get color based on temperature
            temp_color = get_status_color(
                temp,
                {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)}
            )
            
            # Display the forecast card
            st.markdown(f"""
            <div style="padding: 15px; border-radius: 10px; 
                        margin: 5px; background-color: #f8f9fa;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h4 style="margin:0; color: #333;">{day_name}</h4>
                <p style="color:#666; margin:5px 0; font-size:0.8rem;">{date}</p>
                <div style="font-size:1.8rem; text-align:center; margin:5px 0;">
                    {icon_mapping.get(conditions, "🌤️")}
                </div>
                <div style="font-weight:bold; text-align:center; color:{temp_color};">
                    {temp:.1f}°C
                </div>
                <p style="text-align:center; color:#666; font-size:0.7rem; margin:5px 0;">
                    H: {temp_high:.1f}°C<br>L: {temp_low:.1f}°C
                </p>
                <div style="display:flex; justify-content:space-between; margin:5px 0;">
                    <span>☀️</span>
                    <span style="font-weight:bold;">{irradiance:.0f} W/m²</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin:5px 0;">
                    <span>💨</span>
                    <span style="font-weight:bold;">{wind_speed:.1f} m/s</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add expandable details
            with st.expander("Details", expanded=False):
                st.markdown(f"""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: bold;">Cloud Cover:</span>
                        <span>{cloud_cover}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: bold;">Rain Chance:</span>
                        <span>{day.get('rain_probability', 0)*100:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: bold;">Solar Potential:</span>
                        <span style="color: {'#4CAF50' if day.get('solar_potential') == 'High' else '#FFC107' if day.get('solar_potential') == 'Medium' else '#F44336'}">
                            {day.get('solar_potential', 'Medium')}
                        </span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: bold;">Wind Potential:</span>
                        <span style="color: {'#4CAF50' if day.get('wind_potential') == 'High' else '#FFC107' if day.get('wind_potential') == 'Medium' else '#F44336'}">
                            {day.get('wind_potential', 'Medium')}
                        </span>
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
        
        temp = current.get('temperature', 20) + temp_variation
        wind = current.get('wind_speed', 3) + wind_variation
        clouds = current.get('cloud_cover', 50) + cloud_variation
        
        temp = max(min(temp, 40), -5)
        wind = max(min(wind, 20), 0)
        clouds = max(min(clouds, 100), 0)
        
        sunrise = current.get('sunrise')
        sunset = current.get('sunset')
        if sunrise and sunset:
            is_day = sunrise.time() <= date.time() <= sunset.time()
        else:
            is_day = True  # fallback if no sunrise/sunset
        
        irradiance = current.get('irradiance', 500) * (1 - clouds/100) if is_day else 0
        
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
            color=["#FF5252", "#FFD600"]
        )
    
    with hist_tab2:
        st.line_chart(
            chart_data,
            x="date",
            y=["wind_speed", "cloud_cover"],
            color=["#2962FF", "#9E9E9E"]
        )
    
    with hist_tab3:
        st.line_chart(
            chart_data,
            x="date",
            y=["solar_power", "wind_power"],
            color=["#FFD600", "#2962FF"]
        )

# Weather-based Recommendations
st.subheader("Energy Optimization Recommendations")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    st.markdown("### Today's Generation Strategy")
    
    if irradiance is not None and wind_speed is not None:
        if irradiance > 500 and wind_speed < 3:
            strategy = "Solar Priority"
            color = "#FFD600"
        elif irradiance < 300 and wind_speed > 4:
            strategy = "Wind Priority"
            color = "#2962FF"
        else:
            strategy = "Balanced"
            color = "#9C27B0"
    else:
        strategy = "Unknown"
        color = "#9E9E9E"
    
    st.markdown(f"""
    <div style="padding:15px; border-radius:5px; border-left:5px solid {color}; background-color:#f8f9fa;">
        <h3 style="margin:0; color:{color};">{strategy}</h3>
        <p style="margin-top:8px;">{'Focus on solar generation' if strategy == 'Solar Priority' else 
                                  'Focus on wind generation' if strategy == 'Wind Priority' else 
                                  'Balance between solar and wind' if strategy == 'Balanced' else
                                  'Insufficient data'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Action Items")
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
    elif strategy == "Balanced":
        st.markdown("""
        - Monitor both systems
        - Balance energy storage
        - Be prepared to shift focus
        """)
    else:
        st.markdown("- Insufficient data for recommendations.")

with opt_col2:
    st.markdown("### 5-Day Energy Forecast")
    
    dates = [day['date'] for day in forecast[:5]]
    solar_potential = [100 if day['solar_potential'] == "High" else 60 if day['solar_potential'] == "Medium" else 30 for day in forecast[:5]] if forecast else []
    wind_potential = [100 if day['wind_potential'] == "High" else 60 if day['wind_potential'] == "Medium" else 30 for day in forecast[:5]] if forecast else []
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=solar_potential,
        name="Solar",
        marker_color="#FFD600"
    ))
    fig.add_trace(go.Bar(
        x=dates,
        y=wind_potential,
        name="Wind",
        marker_color="#2962FF"
    ))
    fig.update_layout(
        barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if solar_potential:
        best_solar_day = dates[solar_potential.index(max(solar_potential))]
    else:
        best_solar_day = "N/A"

    if wind_potential:
        best_wind_day = dates[wind_potential.index(max(wind_potential))]
    else:
        best_wind_day = "N/A"

    if solar_potential and wind_potential:
        conserve_day = dates[(np.array(solar_potential) + np.array(wind_potential)).argmin()]
    else:
        conserve_day = "N/A"

    st.markdown(f"""
    - **Best solar day:** {best_solar_day}
    - **Best wind day:** {best_wind_day}
    - **Conserve energy on:** {conserve_day}
    """)

# Final Auto-Refresh JavaScript
st.markdown("""
<script>
    // Full page refresh every 4 hours as fallback
    setTimeout(function() {
        window.location.reload();
    }, 14400000);  // 4 hours
</script>
""", unsafe_allow_html=True)