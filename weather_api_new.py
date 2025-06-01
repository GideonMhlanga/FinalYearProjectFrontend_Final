import os
import requests
import logging
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Weather API client for weatherapi.com with solar/wind optimization"""
    
    def __init__(self):
        self.api_key = os.getenv('WEATHERAPI_KEY', '1295574e327a4f1797364201251204')
        self.base_url = "https://api.weatherapi.com/v1"
        self.default_location = "Bulawayo"
        self.timezone = "Africa/Harare"
        self.cache = {}
        self.cache_timeout = 1800  # 30 minute cache (weather changes frequently)
        
        # Available locations in Zimbabwe
        self.available_locations = [
            "Bulawayo", "Harare", "Gweru", "Mutare",
            "Victoria Falls", "Chitungwiza", "Kwekwe",
            "Kadoma", "Masvingo", "Chinhoyi"
        ]

    def _make_api_request(self, endpoint, params=None):
        """Make API request with caching"""
        cache_key = f"{endpoint}-{str(params)}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now(pytz.utc) - cached_time).total_seconds() < self.cache_timeout:
                return cached_data
        
        try:
            # Build request URL
            url = f"{self.base_url}/{endpoint}.json?key={self.api_key}"
            if params:
                url += "&" + "&".join([f"{k}={v}" for k,v in params.items()])
            
            # Make the API call
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            self.cache[cache_key] = (datetime.now(pytz.utc), data)
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            raise Exception(f"Weather API request failed: {str(e)}")

    def _process_current(self, current_data):
        """Extract only relevant current weather data for solar/wind"""
        return {
            "timestamp": datetime.strptime(current_data['last_updated'], '%Y-%m-%d %H:%M'),
            "temp_c": current_data.get('temp_c'),
            "condition": current_data.get('condition', {}).get('text'),
            "wind_kph": current_data.get('wind_kph'),
            "wind_dir": current_data.get('wind_dir'),
            "gust_kph": current_data.get('gust_kph'),
            "cloud": current_data.get('cloud'),
            "uv": current_data.get('uv'),
            "humidity": current_data.get('humidity'),
            "pressure_mb": current_data.get('pressure_mb'),
            "precip_mm": current_data.get('precip_mm'),
            "is_day": current_data.get('is_day'),
            "solar_irradiance": self._estimate_irradiance(
                current_data.get('cloud'),
                current_data.get('uv'),
                current_data.get('is_day') == 1
            )
        }

    def _process_forecast_day(self, forecast_day):
        """Process a single day's forecast data"""
        day_data = forecast_day.get('day', {})
        astro_data = forecast_day.get('astro', {})
        
        return {
            "date": forecast_day.get('date'),
            "max_temp_c": day_data.get('maxtemp_c'),
            "min_temp_c": day_data.get('mintemp_c'),
            "avg_wind_kph": day_data.get('maxwind_kph'),
            "total_precip_mm": day_data.get('totalprecip_mm'),
            "condition": day_data.get('condition', {}).get('text'),
            "daily_chance_of_rain": day_data.get('daily_chance_of_rain'),
            "uv": day_data.get('uv'),
            "sunrise": astro_data.get('sunrise'),
            "sunset": astro_data.get('sunset'),
            "moonrise": astro_data.get('moonrise'),
            "moonset": astro_data.get('moonset')
        }

    def _process_hourly(self, hourly_data):
        """Process hourly forecast data"""
        return [{
            "time": hour.get('time').split()[1],  # Just get HH:MM
            "temp_c": hour.get('temp_c'),
            "wind_kph": hour.get('wind_kph'),
            "gust_kph": hour.get('gust_kph'),
            "cloud": hour.get('cloud'),
            "chance_of_rain": hour.get('chance_of_rain'),
            "condition": hour.get('condition', {}).get('text'),
            "is_day": hour.get('is_day'),
            "solar_potential": self._estimate_irradiance(
                hour.get('cloud'),
                hour.get('uv', 1),  # Default UV of 1 if not available
                hour.get('is_day') == 1
            )
        } for hour in hourly_data]

    def _estimate_irradiance(self, cloud_cover, uv_index, is_daytime):
        """Estimate solar irradiance based on weather conditions"""
        if not is_daytime:
            return 0
        
        cloud_cover = cloud_cover or 0  # Default to 0 if None
        uv_index = uv_index or 1  # Default to 1 if None
        
        # Simple estimation formula (can be refined)
        base_irradiance = 1000  # W/m² for clear sky
        cloud_factor = 1 - (cloud_cover / 100 * 0.7)  # Clouds reduce irradiance
        uv_factor = uv_index / 10  # UV index correlates with irradiance
        
        return max(0, min(base_irradiance, base_irradiance * cloud_factor * uv_factor))

    def get_current_weather(self, location=None):
        """Get current weather data optimized for solar/wind systems"""
        location = location or self.default_location
        
        try:
            data = self._make_api_request("current", {"q": f"{location},ZW"})
            
            if not data.get('current'):
                raise Exception("No current weather data returned from API")
                
            return self._process_current(data['current'])
            
        except Exception as e:
            logging.error(f"Failed to get current weather: {str(e)}")
            raise Exception(f"Failed to get current weather: {str(e)}")

    def get_forecast(self, location=None, days=3):
        """Get forecast data optimized for solar/wind systems"""
        location = location or self.default_location
        
        try:
            data = self._make_api_request("forecast", {
                "q": f"{location},ZW",
                "days": days,
                "aqi": "no",
                "alerts": "no"
            })
            
            if not data.get('forecast'):
                raise Exception("No forecast data returned from API")
                
            forecast_days = data['forecast']['forecastday']
            
            return {
                "location": data['location'],
                "current": self._process_current(data['current']),
                "forecast": [self._process_forecast_day(day) for day in forecast_days],
                "hourly": self._process_hourly(forecast_days[0]['hour']) if forecast_days else []
            }
            
        except Exception as e:
            logging.error(f"Failed to get forecast: {str(e)}")
            raise Exception(f"Failed to get forecast: {str(e)}")

    def get_available_locations(self):
        """Return list of available locations"""
        return self.available_locations

# Initialize API
weather_api = WeatherAPI()

# Check API status
try:
    test_data = weather_api.get_current_weather()
    is_api_working = True
except Exception as e:
    is_api_working = False
    logging.error(f"API test failed: {str(e)}")

# Display status in Streamlit
if not is_api_working:
    st.error("⚠️ Weather API is currently unavailable. Using cached data if available.")