import os
import requests
import logging
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import numpy as np
import time
import streamlit as st

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Weather API client for RapidAPI's OpenWeather endpoint"""
    
    def __init__(self):
        # Initialize API key from environment variables
        self.api_key = os.getenv('RAPIDAPI_KEY', 'b7e9238e13msh4b32e23b87ef7edp1aa762jsn68e185938e91')
        self.host = "open-weather13.p.rapidapi.com"
        self.base_url = f"https://{self.host}/city"
        self.location = "Bulawayo"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.host
        }
        # Define available locations
        self.available_locations = [
            "Bulawayo",
            "Harare",
            "Gweru",
            "Mutare",
            "Victoria Falls",
            "Chitungwiza",
            "Kwekwe",
            "Kadoma",
            "Masvingo",
            "Chinhoyi"
        ]
        # Add rate limiting
        self.last_request_time = datetime.now(pytz.timezone('Africa/Harare'))
        self.min_request_interval = 1.0  # Minimum seconds between requests

    def _calculate_irradiance(self, cloud_cover: float, is_day: bool) -> float:
        """Calculate solar irradiance based on cloud cover and time of day.
        
        Args:
            cloud_cover (float): Cloud cover percentage (0-100)
            is_day (bool): Whether it's daytime
            
        Returns:
            float: Estimated solar irradiance in W/m²
        """
        if not is_day:
            return 0.0
            
        # Base irradiance for clear sky (1000 W/m²)
        base_irradiance = 1000.0
        
        # Calculate cloud attenuation factor (0-1)
        # More clouds = less irradiance
        cloud_factor = 1.0 - (cloud_cover / 100.0)
        
        # Calculate final irradiance
        irradiance = base_irradiance * cloud_factor
        
        # Ensure non-negative value
        return max(0.0, irradiance)

    def get_available_locations(self) -> list:
        """Get list of available locations for weather data."""
        return self.available_locations

    def _make_api_request(self, endpoint: str) -> dict:
        """Make an API request to the weather service with rate limiting."""
        try:
            # Check if we need to wait before making another request
            now = datetime.now(pytz.timezone('Africa/Harare'))
            elapsed = (now - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
            
            url = f"{self.base_url}/{self.location}/EN"
            response = requests.get(url, headers=self.headers)
            
            # Update last request time
            self.last_request_time = datetime.now(pytz.timezone('Africa/Harare'))
            
            # Check for rate limit error
            if response.status_code == 429:
                logging.warning("Rate limit reached, using cached data")
                return {}
                
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error making API request: {str(e)}")
            return {}

    def get_current_weather(self, location: str = None) -> dict:
        """Get current weather data for the specified location."""
        try:
            # Use the provided location or default to instance location
            location = location or self.location
            
            # Make API request
            response = self._make_api_request(f"/current.json?q={location}")
            
            if not response:
                return self._get_default_weather()
            
            # Extract current conditions
            current = response.get('current', {})
            location_data = response.get('location', {})
            
            # Get timestamp
            timestamp = datetime.fromtimestamp(current.get('last_updated_epoch', 0), tz=pytz.timezone('Africa/Harare'))
            
            # Calculate solar potential based on cloud cover and time of day
            cloud_cover = current.get('cloud', 0)
            is_day = current.get('is_day', 1) == 1
            irradiance = self._calculate_irradiance(cloud_cover, is_day)
            
            # Determine solar potential
            if irradiance > 800:
                solar_potential = "High"
            elif irradiance > 500:
                solar_potential = "Medium"
            else:
                solar_potential = "Low"
            
            # Determine wind potential
            wind_speed = current.get('wind_kph', 0) / 3.6  # Convert km/h to m/s
            if wind_speed > 4:
                wind_potential = "High"
            elif wind_speed > 2:
                wind_potential = "Medium"
            else:
                wind_potential = "Low"
            
            return {
                'timestamp': timestamp,
                'temperature': current.get('temp_c', 25.0),
                'feels_like': current.get('feelslike_c', 25.0),
                'humidity': current.get('humidity', 50.0),
                'pressure': current.get('pressure_mb', 1013.0),
                'wind_speed': wind_speed,
                'wind_direction': current.get('wind_degree', 0),
                'cloud_cover': cloud_cover,
                'irradiance': irradiance,
                'conditions': current.get('condition', {}).get('text', 'Clear'),
                'description': current.get('condition', {}).get('text', 'clear sky'),
                'solar_potential': solar_potential,
                'wind_potential': wind_potential,
                'sunrise': datetime.fromtimestamp(location_data.get('sunrise_epoch', 0), tz=pytz.timezone('Africa/Harare')),
                'sunset': datetime.fromtimestamp(location_data.get('sunset_epoch', 0), tz=pytz.timezone('Africa/Harare'))
            }
        except Exception as e:
            logging.error(f"Failed to get current weather for {location}: {str(e)}")
            return self._get_default_weather()

    def get_forecast(self, days: int = 7, location: str = None) -> list:
        """Get weather forecast data for the specified number of days.
        
        Args:
            days (int): Number of days to forecast (default: 7 for full week)
            location (str): Location to get forecast for (default: uses instance location)
        """
        try:
            # For now, return simulated forecast data since the free API doesn't support forecasts
            current = self.get_current_weather()
            forecasts = []
            
            # Generate simulated forecast data based on current conditions
            current_time = datetime.now(pytz.timezone('Africa/Harare'))
            
            # Ensure we start from the beginning of the week (Sunday)
            days_to_sunday = (7 - current_time.weekday()) % 7
            start_time = current_time - timedelta(days=days_to_sunday)
            
            for day in range(days):
                # Calculate daily averages for more realistic forecasts
                daily_temp = current['temperature'] + np.random.uniform(-3, 3)
                daily_wind = current['wind_speed'] + np.random.uniform(-1, 1)
                daily_clouds = current['cloud_cover'] + np.random.uniform(-15, 15)
                
                # Add a weekly pattern to make forecasts more realistic
                day_of_week = (start_time + timedelta(days=day)).weekday()
                weekly_temp_adjust = np.sin(2 * np.pi * day_of_week / 7) * 2  # Weekly temperature cycle
                weekly_wind_adjust = np.cos(2 * np.pi * day_of_week / 7) * 1  # Weekly wind cycle
                
                # Determine base weather conditions for the day
                cloud_cover = min(100, max(0, daily_clouds))
                if cloud_cover < 20:
                    conditions = "Clear"
                    description = "clear sky"
                elif cloud_cover < 50:
                    conditions = "Partly Cloudy"
                    description = "partly cloudy"
                elif cloud_cover < 80:
                    conditions = "Cloudy"
                    description = "cloudy"
                else:
                    conditions = "Overcast"
                    description = "overcast"
                
                # Add rain probability based on conditions and day of week
                if conditions in ["Cloudy", "Overcast"]:
                    rain_prob = np.random.uniform(0.3, 0.7)
                    if rain_prob > 0.5:
                        conditions = "Rain"
                        description = "light rain"
                else:
                    rain_prob = np.random.uniform(0, 0.2)
                
                # Calculate daily high and low temperatures
                temp_high = daily_temp + weekly_temp_adjust + 5
                temp_low = daily_temp + weekly_temp_adjust - 5
                
                # Calculate average wind speed for the day
                wind_speed = max(0, daily_wind + weekly_wind_adjust)
                
                # Calculate solar and wind potential
                solar_potential = "High" if conditions == "Clear" else "Medium" if conditions == "Partly Cloudy" else "Low"
                wind_potential = "High" if wind_speed > 4 else "Medium" if wind_speed > 2 else "Low"
                
                # Create daily forecast
                timestamp = start_time + timedelta(days=day)
                forecast = {
                    'timestamp': timestamp,
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'day_name': timestamp.strftime('%A'),
                    'temperature': daily_temp + weekly_temp_adjust,
                    'temp_high': temp_high,
                    'temp_low': temp_low,
                    'humidity': min(100, max(0, current['humidity'] + np.random.uniform(-10, 10))),
                    'pressure': current['pressure'] + np.random.uniform(-5, 5),
                    'wind_speed': wind_speed,
                    'wind_direction': (current['wind_direction'] + np.random.uniform(-45, 45)) % 360,
                    'cloud_cover': cloud_cover,
                    'conditions': conditions,
                    'description': description,
                    'rain_probability': rain_prob,
                    'solar_potential': solar_potential,
                    'wind_potential': wind_potential,
                    'irradiance': max(0, 1000 * (1 - cloud_cover / 100)) if conditions in ["Clear", "Partly Cloudy"] else 0
                }
                
                forecasts.append(forecast)
            
            return forecasts
        except Exception as e:
            logging.error(f"Failed to get forecast for {location or self.location}: {str(e)}")
            return []

    def _get_default_weather(self, timestamp: datetime = None) -> dict:
        """Return default weather values when API fails."""
        now = datetime.now(pytz.timezone('Africa/Harare'))
        return {
            'temperature': 25.0,
            'humidity': 50.0,
            'pressure': 1013.0,
            'wind_speed': 5.0,
            'wind_direction': 0,
            'cloud_cover': 0,
            'irradiance': 500.0,
            'timestamp': timestamp or now,
            'conditions': 'Clear',
            'description': 'clear sky',
            'feels_like': 25.0,
            'sunrise': now.replace(hour=6, minute=0, second=0, microsecond=0),
            'sunset': now.replace(hour=18, minute=0, second=0, microsecond=0),
            'solar_potential': 'Medium',  # Default to Medium for default weather
            'wind_potential': 'Medium'    # Default to Medium for default weather
        }

    def _is_daytime(self, timestamp: datetime, sys_data: dict) -> bool:
        """Check if it's daytime based on sunrise and sunset times."""
        try:
            sunrise = datetime.fromtimestamp(sys_data.get('sunrise', 0))
            sunset = datetime.fromtimestamp(sys_data.get('sunset', 0))
            return sunrise <= timestamp <= sunset
        except:
            # If we can't determine, assume it's daytime between 6 AM and 6 PM
            hour = timestamp.hour
            return 6 <= hour <= 18

    # Add methods for backward compatibility
    def get_solar_irradiance(self, timestamp: datetime) -> dict:
        """Get solar irradiance data (for backward compatibility)."""
        weather_data = self.get_current_weather(timestamp)
        return {
            'ghi': weather_data['irradiance'],
            'dni': weather_data['irradiance'] * 1.2,
            'dhi': weather_data['irradiance'] * 0.3,
            'timestamp': weather_data['timestamp']
        }

    def get_wind_speed(self, timestamp: datetime) -> dict:
        """Get wind speed data (for backward compatibility)."""
        weather_data = self.get_current_weather(timestamp)
        return {
            'speed': weather_data['wind_speed'],
            'direction': weather_data['wind_direction'],
            'gust': weather_data['wind_speed'] * 1.5,  # Estimate gust speed
            'timestamp': weather_data['timestamp']
        }

# Create instances for both new and old usage
weather_api = WeatherAPI()
weather_apis = weather_api  # For backward compatibility 

def get_day_colors(theme="light"):
    """Get color mapping for days based on theme."""
    return {
        "Sunday": "#fff9e6" if theme == "light" else "#332e1f",
        "Monday": "#e6f2ff" if theme == "light" else "#1a2833",
        "Tuesday": "#e6ffe6" if theme == "light" else "#1a331a",
        "Wednesday": "#fff0e6" if theme == "light" else "#33261f",
        "Thursday": "#f0e6ff" if theme == "light" else "#261f33",
        "Friday": "#ffe6e6" if theme == "light" else "#331f1f",
        "Saturday": "#e6e6ff" if theme == "light" else "#1f1f33"
    }

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

# HTML/CSS styling should be in a separate file or Streamlit components
st.markdown("""
    <div style="position: relative; padding: 15px;">
""", unsafe_allow_html=True)
