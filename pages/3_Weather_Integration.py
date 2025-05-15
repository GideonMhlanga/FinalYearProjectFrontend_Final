import streamlit as st
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Weather API client using Open-Meteo with enhanced UI components"""
    
    def __init__(self):
        # Setup Open-Meteo API with caching
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=retry_session)
        self.timezone = "Africa/Harare"
        
        # Zimbabwe locations with coordinates
        self.locations = {
            "Harare": {"lat": -17.82917, "lon": 31.05222},
            "Bulawayo": {"lat": -20.15, "lon": 28.5833},
            "Mutare": {"lat": -18.9726, "lon": 32.6709},
            "Gweru": {"lat": -19.45, "lon": 29.8167},
            "Kwekwe": {"lat": -18.9281, "lon": 29.8149},
            "Chitungwiza": {"lat": -18.0127, "lon": 31.0755},
            "Victoria Falls": {"lat": -17.9243, "lon": 25.8567},
            "Kadoma": {"lat": -18.3333, "lon": 29.9167},
            "Masvingo": {"lat": -20.0667, "lon": 30.8333},
            "Chinhoyi": {"lat": -17.3667, "lon": 30.2000}
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if "weather_location" not in st.session_state:
            st.session_state.weather_location = os.getenv("LOCATION", "Bulawayo")
        if "last_refresh_time" not in st.session_state:
            st.session_state.last_refresh_time = datetime.now(pytz.timezone(self.timezone))
        
        # Custom CSS
        st.markdown("""
        <style>
            .main { background-color: #f5f5f5; }
            .custom-card {
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }
            .solar-card { background-color: #FFF9C4; border-left: 5px solid #FFC107; }
            .wind-card { background-color: #B3E5FC; border-left: 5px solid #2196F3; }
            .temp-card { background-color: #FFCCBC; border-left: 5px solid #FF5722; }
            .forecast-card {
                padding: 12px;
                border-radius: 10px;
                margin: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .metric-row { display: flex; justify-content: space-between; margin-bottom: 10px; }
            .metric-box { flex: 1; padding: 0 5px; }
            .metric-title { font-weight: bold; font-size: 0.8rem; color: #555; }
            .metric-value { font-size: 1rem; font-weight: bold; color: #222; }
            .technical-details { 
                background-color: #f0f2f6; 
                padding: 15px; 
                border-radius: 5px; 
                margin-top: 10px;
            }
        </style>
        """, unsafe_allow_html=True)
        
    def get_available_locations(self):
        """Return list of available locations"""
        return list(self.locations.keys())
    
    def get_current_weather(self, location):
        """Get current weather with same structure as original"""
        try:
            return self._get_openmeteo_current(location)
        except Exception as e:
            self.logger.error(f"Failed to get current weather: {str(e)}")
            return self._get_default_current(location)
    
    def get_forecast(self, location):
        """Get forecast with same structure as original"""
        try:
            return self._get_openmeteo_forecast(location)
        except Exception as e:
            self.logger.error(f"Failed to get forecast: {str(e)}")
            return self._get_default_forecast(location)
    
    def _get_openmeteo_current(self, location):
        """Get current weather from Open-Meteo matching original structure"""
        if location not in self.locations:
            location = "Bulawayo"
        
        coords = self.locations[location]
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "current": [
                "temperature_2m", 
                "relative_humidity_2m",
                "apparent_temperature",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code",
                "pressure_msl",
                "cloud_cover",
                "is_day",
                "precipitation",
                "rain",
                "showers",
                "snowfall"
            ],
            "hourly": ["shortwave_radiation"],
            "timezone": self.timezone,
            "wind_speed_unit": "ms"
        }
        
        response = self.openmeteo.weather_api(url, params=params)[0]
        current = response.Current()
        
        # Get hourly irradiance (use first value)
        hourly = response.Hourly()
        irradiance = hourly.Variables(0).ValuesAsNumpy()[0] if len(hourly.Variables(0).ValuesAsNumpy()) > 0 else 0
        
        # Process weather data to match original structure
        weather_code = current.Variables(5).Value()
        is_day = current.Variables(8).Value()
        
        return {
            "timestamp": datetime.now(pytz.timezone(self.timezone)),
            "location": location,
            "temperature": current.Variables(0).Value(),
            "humidity": current.Variables(1).Value(),
            "feels_like": current.Variables(2).Value(),
            "wind_speed": current.Variables(3).Value(),
            "wind_direction": current.Variables(4).Value(),
            "weather_code": weather_code,
            "conditions": self._map_weather_code(weather_code),
            "description": self._get_weather_description(weather_code),
            "pressure": current.Variables(6).Value(),
            "cloud_cover": current.Variables(7).Value(),
            "is_day": is_day,
            "irradiance": irradiance,
            "precipitation": current.Variables(9).Value(),
            "rain": current.Variables(10).Value(),
            "snowfall": current.Variables(11).Value(),
            "solar_potential": self._get_solar_potential(irradiance, is_day),
            "wind_potential": self._get_wind_potential(current.Variables(3).Value())
        }
    
    def _get_openmeteo_forecast(self, location):
        """Get forecast from Open-Meteo matching original structure"""
        if location not in self.locations:
            location = "Bulawayo"
            
        coords = self.locations[location]
        
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "sunrise",
                "sunset",
                "uv_index_max",
                "precipitation_sum",
                "rain_sum",
                "showers_sum",
                "precipitation_probability_max",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant"
            ],
            "timezone": self.timezone,
            "forecast_days": 7
        }
        
        response = self.openmeteo.weather_api(url, params=params)[0]
        daily = response.Daily()
        
        forecast = []
        time_range = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        
        for i in range(7):
            weather_code = daily.Variables(0).ValuesAsNumpy()[i]
            forecast.append({
                "date": time_range[i].strftime("%Y-%m-%d"),
                "day_name": time_range[i].strftime("%A"),
                "temperature": (daily.Variables(1).ValuesAsNumpy()[i] + daily.Variables(2).ValuesAsNumpy()[i]) / 2,
                "temp_high": daily.Variables(1).ValuesAsNumpy()[i],
                "temp_low": daily.Variables(2).ValuesAsNumpy()[i],
                "feels_like_high": daily.Variables(3).ValuesAsNumpy()[i],
                "feels_like_low": daily.Variables(4).ValuesAsNumpy()[i],
                "sunrise": pd.to_datetime(daily.Variables(5).ValuesAsNumpy()[i], unit="s", utc=True),
                "sunset": pd.to_datetime(daily.Variables(6).ValuesAsNumpy()[i], unit="s", utc=True),
                "uv_index": daily.Variables(7).ValuesAsNumpy()[i],
                "precipitation": daily.Variables(8).ValuesAsNumpy()[i],
                "rain": daily.Variables(9).ValuesAsNumpy()[i],
                "showers": daily.Variables(10).ValuesAsNumpy()[i],
                "rain_probability": daily.Variables(11).ValuesAsNumpy()[i] / 100,
                "wind_speed": daily.Variables(12).ValuesAsNumpy()[i],
                "wind_direction": daily.Variables(13).ValuesAsNumpy()[i],
                "weather_code": weather_code,
                "conditions": self._map_weather_code(weather_code),
                "description": self._get_weather_description(weather_code),
                "solar_potential": self._get_solar_potential(daily.Variables(7).ValuesAsNumpy()[i], 1),
                "wind_potential": self._get_wind_potential(daily.Variables(12).ValuesAsNumpy()[i]),
                "cloud_cover": self._estimate_cloud_cover(weather_code)
            })
        
        return forecast
    
    def _map_weather_code(self, code):
        """Map WMO weather codes to conditions"""
        weather_mapping = {
            0: "Clear",
            1: "Mainly Clear", 
            2: "Partly Cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Fog",
            51: "Drizzle",
            53: "Drizzle",
            55: "Drizzle",
            56: "Freezing Drizzle",
            57: "Freezing Drizzle",
            61: "Rain",
            63: "Rain",
            65: "Heavy Rain",
            66: "Freezing Rain",
            67: "Freezing Rain",
            71: "Snow",
            73: "Snow",
            75: "Heavy Snow",
            77: "Snow Grains",
            80: "Rain Showers",
            81: "Rain Showers",
            82: "Heavy Rain Showers",
            85: "Snow Showers",
            86: "Snow Showers",
            95: "Thunderstorm",
            96: "Thunderstorm",
            99: "Thunderstorm"
        }
        return weather_mapping.get(code, "Unknown")
    
    def _get_weather_description(self, code):
        """Get detailed weather description"""
        descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail"
        }
        return descriptions.get(code, "Unknown weather conditions")
    
    def _get_solar_potential(self, irradiance, is_day):
        """Calculate solar potential"""
        if not is_day:
            return "None"
        if irradiance > 700:
            return "High"
        elif irradiance > 400:
            return "Medium"
        else:
            return "Low"
    
    def _get_wind_potential(self, wind_speed):
        """Calculate wind potential"""
        if wind_speed > 6:
            return "High"
        elif wind_speed > 4:
            return "Medium"
        else:
            return "Low"
    
    def _estimate_cloud_cover(self, weather_code):
        """Estimate cloud cover"""
        if weather_code == 0:
            return 0
        elif weather_code == 1:
            return 25
        elif weather_code == 2:
            return 50
        elif weather_code == 3:
            return 100
        elif weather_code in [45, 48]:
            return 90
        else:
            return 70
    
    def _get_default_current(self, location):
        """Default current weather data"""
        return {
            "timestamp": datetime.now(pytz.timezone(self.timezone)),
            "location": location,
            "temperature": 25.0,
            "humidity": 50.0,
            "feels_like": 26.0,
            "wind_speed": 5.0,
            "wind_direction": 0,
            "weather_code": 0,
            "conditions": "Clear",
            "description": "Clear sky",
            "pressure": 1013.0,
            "cloud_cover": 0,
            "is_day": 1,
            "irradiance": 500,
            "precipitation": 0,
            "rain": 0,
            "snowfall": 0,
            "solar_potential": "Medium",
            "wind_potential": "Medium"
        }
    
    def _get_default_forecast(self, location):
        """Get actual forecast data from OpenMeteo API as fallback"""
        try:
            # Try to get real data from the API
            return self._get_openmeteo_forecast(location)
        except Exception as e:
            # If API call fails, provide minimal fallback data
            self.logger.error(f"Failed to get default forecast from API: {str(e)}")
            forecast = []
            today = datetime.now(pytz.timezone(self.timezone))
            
            for i in range(7):
                date = today + timedelta(days=i)
                forecast.append({
                    "timestamp": datetime.now(pytz.timezone(self.timezone)),
                    "location": location,
                    "date": date.strftime("%Y-%m-%d"),
                    "day_name": date.strftime("%A"),
                    "temperature": 25.0,
                    "temp_high": 28.0,
                    "temp_low": 22.0,
                    "feels_like_high": 29.0,
                    "feels_like_low": 21.0,
                    "sunrise": pd.to_datetime(date.replace(hour=6, minute=0, second=0)),
                    "sunset": pd.to_datetime(date.replace(hour=18, minute=0, second=0)),
                    "uv_index": 7.0,
                    "precipitation": 0,
                    "rain": 0,
                    "showers": 0,
                    "rain_probability": 0.1,
                    "wind_speed": 5.0,
                    "wind_direction": 0,
                    "weather_code": 0,
                    "conditions": "Clear",
                    "description": "Clear sky",
                    "solar_potential": "Medium",
                    "wind_potential": "Medium",
                    "cloud_cover": 0
                })
            
            return forecast
    
    def display_weather_ui(self):
        """Display all weather visualizations and cards"""
        st.title("Zimbabwe Weather Integration")
        st.write("Live weather data to optimize your hybrid energy generation")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Settings")
            
            # Auto-refresh configuration
            auto_refresh = st.checkbox("Auto-refresh data", value=True)
            refresh_interval = st.slider("Auto-refresh interval (sec)", 5, 60, 15)
            
            if st.button("Refresh Data Now"):
                st.session_state.last_refresh_time = datetime.now(pytz.timezone(self.timezone))
                st.rerun()
            
            # Location selector
            available_locations = self.get_available_locations()
            selected_location = st.selectbox(
                "Select Location",
                available_locations,
                index=available_locations.index(st.session_state.weather_location)
            )
            
            if selected_location != st.session_state.weather_location:
                st.session_state.weather_location = selected_location
                st.rerun()
            
            # Technical details toggle
            st.session_state.show_technical = st.checkbox("Show technical details", value=False)
            
            # Display last refresh time
            st.caption(f"Last updated: {st.session_state.last_refresh_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get data
        location = st.session_state.get("weather_location", "Bulawayo")
        current = self.get_current_weather(location)
        forecast = self.get_forecast(location)
        
        # Current Weather Cards
        st.subheader("Current Weather Conditions")
        col1, col2, col3 = st.columns(3)
        
        # Solar Card
        with col1:
            irradiance = current.get('irradiance', 0)
            irradiance_color = self._get_status_color(
                irradiance, 
                {"green": (600, float('inf')), "yellow": (200, 600), "red": (0, 200)}
            )
            st.markdown(f"""
            <div class="custom-card solar-card">
                <h3>☀️ Solar Irradiance</h3>
                <h2 style='color: {irradiance_color}'>{irradiance:.1f} W/m²</h2>
                <p>{current.get('solar_potential', 'Medium')} solar potential</p>
                <p style='color: #666; font-size: 0.9rem;'>{current.get('conditions', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get("show_technical", False):
                with st.expander("Technical Details"):
                    st.markdown(f"""
                    <div class="technical-details">
                        <p><strong>Cloud Cover:</strong> {current.get('cloud_cover', 0)}%</p>
                        <p><strong>Sunrise:</strong> {current.get('sunrise', 'N/A')}</p>
                        <p><strong>Sunset:</strong> {current.get('sunset', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Wind Card
        with col2:
            wind_speed = current.get('wind_speed', 0)
            wind_color = self._get_status_color(
                wind_speed,
                {"green": (4, float('inf')), "yellow": (2, 4), "red": (0, 2)}
            )
            st.markdown(f"""
            <div class="custom-card wind-card">
                <h3>💨 Wind Speed</h3>
                <h2 style='color: {wind_color}'>{wind_speed:.1f} m/s</h2>
                <p>{current.get('wind_potential', 'Medium')} wind potential</p>
                <p style='color: #666; font-size: 0.9rem;'>Direction: {current.get('wind_direction', 'N/A')}°</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get("show_technical", False):
                with st.expander("Technical Details"):
                    st.markdown(f"""
                    <div class="technical-details">
                        <p><strong>Wind Gust:</strong> {current.get('wind_gust', 'N/A')} m/s</p>
                        <p><strong>Pressure:</strong> {current.get('pressure', 'N/A')} hPa</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Temperature Card
        with col3:
            temperature = current.get('temperature', 20)
            temp_color = self._get_status_color(
                temperature,
                {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)}
            )
            st.markdown(f"""
            <div class="custom-card temp-card">
                <h3>🌡️ Temperature</h3>
                <h2 style='color: {temp_color}'>{temperature:.1f}°C</h2>
                <p>{'Optimal' if 15 <= temperature <= 25 else 'Too hot' if temperature > 25 else 'Too cold'} for system performance</p>
                <p style='color: #666; font-size: 0.9rem;'>Feels like: {current.get('feels_like', temperature):.1f}°C</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.get("show_technical", False):
                with st.expander("Technical Details"):
                    st.markdown(f"""
                    <div class="technical-details">
                        <p><strong>Humidity:</strong> {current.get('humidity', 'N/A')}%</p>
                        <p><strong>Dew Point:</strong> {current.get('dew_point', 'N/A')}°C</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # 7-Day Forecast
        st.subheader("7-Day Weather Forecast")
        st.markdown("---")
        
        icon_mapping = {
            "Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", 
            "Drizzle": "🌦️", "Thunderstorm": "⛈️", 
            "Snow": "❄️", "Mist": "🌫️", "Overcast": "☁️",
            "Partly Cloudy": "⛅"
        }
        
        if not forecast or not isinstance(forecast, list):
            st.warning("No forecast data available")
        else:
            forecast_cols = st.columns(7)
            today = datetime.now(pytz.timezone(self.timezone)).strftime("%Y-%m-%d")
            
            for i, day in enumerate(forecast[:7]):
                with forecast_cols[i]:
                    weather_code = day.get('weather_code', 0)
                    temp = day.get('temperature', 20)
                    temp_color = self._get_status_color(
                        temp,
                        {"green": (15, 25), "yellow": (5, 15), "red": (25, 45)}
                    )
                    
                    # Highlight today's card
                    is_today = day['date'] == today
                    card_style = "border: 2px solid #4CAF50;" if is_today else ""
                    
                    st.markdown(f"""
                    <div style="padding:15px; border-radius:10px; margin:5px; background-color:#f8f9fa; box-shadow:0 2px 5px rgba(0,0,0,0.1); {card_style}">
                        <h4 style="margin:0; color:#333;">{day.get('day_name', 'Day')}</h4>
                        <p style="color:#666; margin:5px 0; font-size:0.8rem;">{day.get('date', '')}</p>
                        <div style="font-size:1.8rem; text-align:center; margin:5px 0;">
                            {icon_mapping.get(day.get('conditions', 'Clear'), "🌤️")}
                        </div>
                        <div style="font-weight:bold; text-align:center; color:{temp_color};">
                            {temp:.1f}°C
                        </div>
                        <p style="text-align:center; color:#666; font-size:0.7rem; margin:5px 0;">
                            H: {day.get('temp_high', temp + 5):.1f}°C<br>L: {day.get('temp_low', temp - 5):.1f}°C
                        </p>
                        <div style="display:flex; justify-content:space-between; margin:5px 0;">
                            <span>☀️</span>
                            <span style="font-weight:bold;">{day.get('irradiance', 0):.0f} W/m²</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin:5px 0;">
                            <span>💨</span>
                            <span style="font-weight:bold;">{day.get('wind_speed', 0):.1f} m/s</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Details", expanded=False):
                        st.markdown(f"""
                        <div style="padding:10px; background-color:#f0f2f6; border-radius:5px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="font-weight:bold;">Cloud Cover:</span>
                                <span>{day.get('cloud_cover', 0)}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="font-weight:bold;">Rain Chance:</span>
                                <span>{day.get('rain_probability', 0)*100:.0f}%</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                                <span style="font-weight:bold;">Solar Potential:</span>
                                <span style="color:{'#4CAF50' if day.get('solar_potential') == 'High' else '#FFC107' if day.get('solar_potential') == 'Medium' else '#F44336'}">
                                    {day.get('solar_potential', 'Medium')}
                                </span>
                            </div>
                            <div style="display:flex; justify-content:space-between;">
                                <span style="font-weight:bold;">Wind Potential:</span>
                                <span style="color:{'#4CAF50' if day.get('wind_potential') == 'High' else '#FFC107' if day.get('wind_potential') == 'Medium' else '#F44336'}">
                                    {day.get('wind_potential', 'Medium')}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Historical Data Visualization
        st.subheader("Historical Weather Data")
        st.markdown("---")
        
        historical_data = self._generate_historical_data(current, forecast)
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
            
            if irradiance > 500 and wind_speed < 3:
                strategy = "Solar Priority"
                color = "#FFD600"
            elif irradiance < 300 and wind_speed > 4:
                strategy = "Wind Priority"
                color = "#2962FF"
            else:
                strategy = "Balanced"
                color = "#9C27B0"
            
            st.markdown(f"""
            <div style="padding:15px; border-radius:5px; border-left:5px solid {color}; background-color:#f8f9fa;">
                <h3 style="margin:0; color:{color};">{strategy}</h3>
                <p style="margin-top:8px;">{'Focus on solar generation' if strategy == 'Solar Priority' else 
                                        'Focus on wind generation' if strategy == 'Wind Priority' else 
                                        'Balance between solar and wind'}</p>
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
            else:
                st.markdown("""
                - Monitor both systems
                - Balance energy storage
                - Be prepared to shift focus
                """)
        
        with opt_col2:
            st.markdown("### 5-Day Energy Forecast")
            
            dates = [day['date'] for day in forecast[:5]] if forecast else []
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
            
            best_solar_day = dates[solar_potential.index(max(solar_potential))] if solar_potential else "N/A"
            best_wind_day = dates[wind_potential.index(max(wind_potential))] if wind_potential else "N/A"
            conserve_day = dates[(np.array(solar_potential) + np.array(wind_potential)).argmin()] if (solar_potential and wind_potential) else "N/A"
            
            st.markdown(f"""
            - **Best solar day:** {best_solar_day}
            - **Best wind day:** {best_wind_day}
            - **Conserve energy on:** {conserve_day}
            """)
    
    def _generate_historical_data(self, current, forecast):
        """Generate simulated historical data"""
        dates = pd.date_range(end=datetime.now(pytz.timezone(self.timezone)), periods=30, freq="D")
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
            
            is_day = current.get('sunrise', datetime.now(pytz.timezone(self.timezone))).time() <= date.time() <= current.get('sunset', datetime.now(pytz.timezone(self.timezone))).time()
            irradiance = current.get('irradiance', 500) * (1 - clouds/100) if is_day else 0
            
            solar_power = max(0, irradiance * 0.15 * (1 + np.random.normal(0, 0.1)))
            wind_power = max(0, wind**3 * 0.5 * (1 + np.random.normal(0, 0.2)))
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "temperature": temp,
                "wind_speed": wind,
                "cloud_cover": clouds,
                "irradiance": irradiance,
                "solar_power": solar_power,
                "wind_power": wind_power
            })
        
        return pd.DataFrame(data)
    
    def _get_status_color(self, value, thresholds):
        """Determine status color based on thresholds"""
        try:
            if isinstance(value, str):
                value = float(value)
            if value is None:
                return "#9E9E9E"  # Grey for missing data
                
            if value >= thresholds["green"][0]:
                return "#4CAF50"  # Green
            elif value >= thresholds["yellow"][0]:
                return "#FFC107"  # Yellow
            else:
                return "#F44336"  # Red
        except (ValueError, TypeError):
            return "#9E9E9E"

# Initialize and run the app
if __name__ == "__main__":
    weather_api = WeatherAPI()
    
    # Check API status
    is_api_working = True
    try:
        test_data = weather_api.get_current_weather("Bulawayo")
        if not test_data:
            is_api_working = False
    except Exception as e:
        is_api_working = False
        logging.error(f"API test failed: {str(e)}")
    
    # Display status
    if not is_api_working:
        st.error("⚠️ Weather API is currently offline. Displaying simulated data.")
    
    # Display the UI
    weather_api.display_weather_ui()