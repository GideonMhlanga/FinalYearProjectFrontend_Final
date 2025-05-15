import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Weather API client using Open-Meteo"""
    
    def __init__(self):
        # Setup Open-Meteo API
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
        
    def get_available_locations(self):
        """Return list of available locations"""
        return list(self.locations.keys())
    
    def get_current_weather(self, location):
        """Get current weather"""
        try:
            return self._get_openmeteo_current(location)
        except Exception as e:
            logging.warning(f"Open-Meteo failed: {str(e)}")
            return self._get_default_current(location)
    
    def get_forecast(self, location):
        """Get forecast"""
        try:
            return self._get_openmeteo_forecast(location)
        except Exception as e:
            logging.warning(f"Open-Meteo forecast failed: {str(e)}")
            return self._get_default_forecast(location)
    
    def _get_openmeteo_current(self, location):
        """Get current weather from Open-Meteo"""
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
            "hourly": ["global_tilted_irradiance_instant"],
            "timezone": self.timezone,
            "wind_speed_unit": "ms"
        }
        
        response = self.openmeteo.weather_api(url, params=params)[0]
        current = response.Current()
        
        # Get hourly irradiance (use first value)
        hourly = response.Hourly()
        irradiance = hourly.Variables(0).ValuesAsNumpy()[0] if len(hourly.Variables(0).ValuesAsNumpy()) > 0 else 0
        
        # Process weather data
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
        """Get forecast from Open-Meteo"""
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
        """Calculate solar potential based on irradiance and daylight"""
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
        """Estimate cloud cover based on weather code"""
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
        """Default forecast data"""
        forecast = []
        today = datetime.now(pytz.timezone(self.timezone))
        
        for i in range(7):
            date = today + timedelta(days=i)
            forecast.append({
                "date": date.strftime("%Y-%m-%d"),
                "day_name": date.strftime("%A"),
                "temperature": 25.0,
                "temp_high": 28.0,
                "temp_low": 22.0,
                "feels_like_high": 29.0,
                "feels_like_low": 21.0,
                "sunrise": "06:00",
                "sunset": "18:00",
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

# Initialize API
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
    import streamlit as st
    st.error("⚠️ Weather API is currently offline. Displaying simulated data.")