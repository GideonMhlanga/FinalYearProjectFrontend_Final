import streamlit as st
from datetime import datetime, timedelta
import pytz
import os
from weather_api_new import WeatherAPI
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go

# Configure the page
st.set_page_config(
    page_title="Solar-Wind Hybrid Monitor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Timezone setup
tz = pytz.timezone("Africa/Harare")

# Initialize WeatherAPI
weather_api = WeatherAPI()

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "weather_location" not in st.session_state:
    st.session_state.weather_location = os.getenv("LOCATION", "Bulawayo")
if "last_refresh_time" not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(tz)
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None

# Custom CSS
st.markdown("""
<style>
    .custom-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .solar-card {
        background-color: #FFF9C4;
    }
    .wind-card {
        background-color: #E3F2FD;
    }
    .temp-card {
        background-color: #FFEBEE;
    }
    .forecast-card {
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_status_color(value, thresholds):
    if value >= thresholds["green"][0]:
        return "#4CAF50"
    elif value >= thresholds["yellow"][0]:
        return "#FFC107"
    else:
        return "#F44336"

def refresh_weather_data():
    try:
        weather_data = weather_api.get_forecast(st.session_state.weather_location, days=7)
        
        # Process current weather
        current = weather_data['current']
        current['irradiance'] = current['solar_irradiance']
        current['wind_speed'] = current['wind_kph'] / 3.6  # Convert km/h to m/s
        current['temperature'] = current['temp_c']
        current['weather_summary'] = current['condition']
        
        # Process daily forecast
        daily_forecast = []
        for day in weather_data['forecast']:
            daily_forecast.append({
                'date': day['date'],
                'day_name': datetime.strptime(day['date'], '%Y-%m-%d').strftime('%A'),
                'temperature': (day['max_temp_c'] + day['min_temp_c']) / 2,
                'max_temp_c': day['max_temp_c'],
                'min_temp_c': day['min_temp_c'],
                'wind_speed': day['avg_wind_kph'] / 3.6,  # Convert km/h to m/s
                #'wind_dir': None,  # Not available in daily forecast
                #'cloud_cover': None,  # Not available in daily forecast
                #'humidity': None,  # Not available in daily forecast
                'weather_summary': day['condition'],
                'weather_icon': None,  # Will map condition to icon later
                'irradiance': day['uv'] * 100,  # Approximate conversion from UV index
                'precipitation': day['total_precip_mm'],
                'solar_potential': day['uv'] * 100 * 0.5,
                'wind_potential': (day['avg_wind_kph'] / 3.6) ** 3 * 0.5
            })
        
        # Process hourly forecast
        hourly_forecast = []
        for hour in weather_data['hourly']:
            hourly_forecast.append({
                'time': hour['time'],
                'temperature': hour['temp_c'],
                'wind_speed': hour['wind_kph'] / 3.6,
                #'wind_dir': hour['wind_dir'],
                'cloud_cover': hour['cloud'],
                #'humidity': None,  # Not available in hourly forecast
                'weather_summary': hour['condition'],
                'weather_icon': None,  # Will map condition to icon later
                'irradiance': hour['solar_potential'],
                'precipitation': hour['chance_of_rain']
            })
        
        return {
            "current": current,
            "daily_forecast": daily_forecast,
            "hourly_forecast": hourly_forecast,
            "timestamp": datetime.now(tz)
        }
    except Exception as e:
        st.error(f"Failed to fetch weather data: {str(e)}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    
    auto_refresh = st.checkbox("Auto-refresh data", value=True)
    refresh_interval = st.slider("Auto-refresh interval (sec)", 5, 60, 15)
    
    if st.button("Refresh Data Now"):
        st.session_state.weather_data = refresh_weather_data()
        st.session_state.last_refresh_time = datetime.now(tz)
    
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
    
    with st.expander("API Settings"):
        api_key = st.text_input(
            "WeatherAPI Key",
            value=os.getenv("WEATHERAPI_KEY", ""),
            type="password"
        )
        if api_key and api_key != os.getenv("WEATHERAPI_KEY"):
            os.environ["WEATHERAPI_KEY"] = api_key
            weather_api.api_key = api_key
            st.session_state.weather_data = refresh_weather_data()

# Auto-refresh logic
elapsed = (datetime.now(tz) - st.session_state.last_refresh_time).total_seconds()
if auto_refresh and elapsed >= refresh_interval:
    st.session_state.weather_data = refresh_weather_data()
    st.session_state.last_refresh_time = datetime.now(tz)

# Initialize data
if st.session_state.weather_data is None:
    st.session_state.weather_data = refresh_weather_data()

if st.session_state.weather_data is None:
    st.error("Failed to load weather data. Please try again later.")
    st.stop()

current_weather = st.session_state.weather_data["current"]
daily_forecast = st.session_state.weather_data["daily_forecast"]
hourly_forecast = st.session_state.weather_data["hourly_forecast"]
last_refresh = st.session_state.last_refresh_time

# Main dashboard
st.title("Zimbabwe Weather Integration")
st.write("Live Zimbabwe weather data and forecasts to optimize your energy generation")

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
        <p>{current_weather['weather_summary']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="custom-card wind-card">
        <h3>💨 Wind Speed</h3>
        <h2 style='color: {get_status_color(current_weather['wind_speed'], {"green": (4, float('inf')), "yellow": (2, 4), "red": (0, 2)})}'>
            {current_weather['wind_speed']:.1f} m/s
        </h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="custom-card temp-card">
        <h3>🌡️ Temperature</h3>
        <h2 style='color: {get_status_color(current_weather['temperature'], {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)})}'>
            {current_weather['temperature']:.1f}°C
        </h2>
        <p>Humidity: {current_weather['humidity'] or 'N/A'}%</p>
    </div>
    """, unsafe_allow_html=True)

# 3-Day Weather Forecast Section
st.subheader("3-Day Weather Forecast")
st.markdown("---")

# Custom CSS
st.markdown("""
<style>
    .forecast-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 20px;
    }
    .forecast-day {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }
    .forecast-date {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 15px;
    }
    .forecast-icon {
        font-size: 2.5rem;
        text-align: center;
        margin: 15px 0;
    }
    .forecast-temp {
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin: 10px 0;
    }
    .forecast-divider {
        height: 1px;
        background: #eee;
        margin: 15px 0;
    }
    .forecast-detail {
        display: flex;
        justify-content: space-between;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Create columns
col1, col2, col3 = st.columns(3)

# Icon mapping
icon_map = {
    'Sunny': '☀️',
    'Clear': '🌙',
    'Partly cloudy': '⛅',
    'Cloudy': '☁️',
    'Overcast': '☁️',
    'Mist': '🌫️',
    'Rain': '🌧️',
    'Thunderstorm': '⛈️',
    'Snow': '❄️'
}

# Forecast card function
def create_forecast_card(day):
    weather_icon = icon_map.get(day['weather_summary'], '🌤️')
    return f"""
    <div class="forecast-card">
        <div>
            <p class="forecast-day">{day['day_name']}</p>
            <p class="forecast-date">{day['date']}</p>
        </div>
        <div class="forecast-icon">{weather_icon}</div>
        <div class="forecast-temp">{day['max_temp_c']:.0f}° / {day['min_temp_c']:.0f}°C</div>
        <div class="forecast-divider"></div>
        <div class="forecast-detail">
            <span>☀️ Irradiance</span>
            <span>{day['irradiance']:.0f} W/m²</span>
        </div>
        <div class="forecast-detail">
            <span>💨 Wind</span>
            <span>{day['wind_speed']:.1f} m/s</span>
        </div>
        <div class="forecast-detail">
            <span>💧 Precip</span>
            <span>{day['precipitation']:.1f} mm</span>
        </div>
    </div>
    """

# Display cards
with col1:
    st.markdown(create_forecast_card(daily_forecast[0]), unsafe_allow_html=True)
with col2:
    st.markdown(create_forecast_card(daily_forecast[1]), unsafe_allow_html=True)
with col3:
    st.markdown(create_forecast_card(daily_forecast[2]), unsafe_allow_html=True)

# Historical Data
st.subheader("Historical Weather Data")
st.markdown("---")

hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Temperature & Irradiance", "Wind Data", "Energy Generation"])

with hist_tab1:
    df_temp_irradiance = pd.DataFrame(daily_forecast[:7])
    df_temp_irradiance['date'] = pd.to_datetime(df_temp_irradiance['date'])
    st.line_chart(
        df_temp_irradiance,
        x="date",
        y=["temperature", "irradiance"],
        color=["#FF5252", "#FFD600"]
    )

with hist_tab2:
    df_wind = pd.DataFrame(daily_forecast[:7])
    df_wind['date'] = pd.to_datetime(df_wind['date'])
    st.line_chart(
        df_wind,
        x="date",
        y=["wind_speed"],
        color=["#2962FF"]
    )

with hist_tab3:
    df_energy = pd.DataFrame(daily_forecast[:7])
    df_energy['date'] = pd.to_datetime(df_energy['date'])
    st.line_chart(
        df_energy,
        x="date",
        y=["solar_potential", "wind_potential"],
        color=["#FFD600", "#2962FF"]
    )

# Weather-based optimization
st.subheader("Weather-Based Energy Optimization")
opt_col1, opt_col2 = st.columns(2)

with opt_col1:
    st.markdown("### Today's Generation Strategy")
    strategy = "Solar Priority" if current_weather['irradiance'] > 500 else "Wind Priority" if current_weather['wind_speed'] > 4 else "Balanced"
    st.markdown(f"""
    <div style="padding:15px; border-radius:5px; background-color:#E8F5E9;">
        <h3>{strategy}</h3>
        <p>Recommended based on current conditions:</p>
        <ul>
            <li>Solar: {current_weather['irradiance']:.1f} W/m²</li>
            <li>Wind: {current_weather['wind_speed']:.1f} m/s</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with opt_col2:
    st.markdown("### 5-Day Energy Forecast")
    df_forecast = pd.DataFrame(daily_forecast[:5])
    df_forecast['date'] = pd.to_datetime(df_forecast['date'])
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_forecast['date'],
        y=df_forecast['solar_potential'],
        name="Solar Potential"
    ))
    fig.add_trace(go.Bar(
        x=df_forecast['date'],
        y=df_forecast['wind_potential'],
        name="Wind Potential"
    ))
    fig.update_layout(barmode='group')
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh JavaScript
st.markdown("""
<script>
    setTimeout(function() {
        window.location.reload();
    }, 300000);
</script>
""", unsafe_allow_html=True)