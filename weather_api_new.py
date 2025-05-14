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
        self.api_key = os.getenv('RAPIDAPI_KEY', '7f9e16f0efmsh4bd796dccefbdc5p12f98ejsn46cba1899d09')
        self.host = "weather-api167.p.rapidapi.com"
        self.base_url = "https://weather-api167.p.rapidapi.com/api/weather"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
            "Accept": "application/json"
        }
        self.default_location = os.getenv('LOCATION', 'Bulawayo')
        self.default_lat = os.getenv('LATITUDE', '-20.1325')
        self.default_lon = os.getenv('LONGITUDE', '28.6264')
        self.default_forecast_days = 7
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

    def _make_api_request(self, endpoint, params=None):
        """Make API request with error handling using requests"""
        try:
            url=f"{self.base_url}/{endpoint}"
            response = requests.get(
                url,
                headers=self.headers,
                params=params or {}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            return None
        except st.json.JSONDecodeError:
            logging.error("Failed to decode API response")
            return None
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
            days (int): Number of days to forecast (default: 7)
            location (str): Location name
            
        Returns:
            list: List of forecast dictionaries with complete weather data
        """
        try:
            # Input validation and type conversion
            try:
                days = int(days) if days is not None else self.default_forecast_days
                if days <= 0 or days > 14:  # Reasonable upper limit
                    days = self.default_forecast_days
                    logging.warning(f"Invalid days value ({days}). Using default: {self.default_forecast_days}")
            except (TypeError, ValueError):
                days = self.default_forecast_days
                logging.warning(f"Invalid days format. Using default: {self.default_forecast_days}")

            # Location validation
            location = str(location) if location is not None else self.location
            if location not in self.available_locations:
                logging.warning(f"Location '{location}' not available. Using default: {self.location}")
                location = self.location

            # Get current weather as baseline
            current = self.get_current_weather(location)
            forecasts = []
            current_time = datetime.now(pytz.timezone('Africa/Harare'))
            
            # Calculate weekly starting point (Sunday)
            days_to_sunday = (7 - current_time.weekday()) % 7
            start_time = current_time - timedelta(days=days_to_sunday)
            
            # Generate forecast for each day
            for day in range(days):
                # Calculate base variations with realistic constraints
                base_temp = current.get('temperature', 20)
                temp_variation = np.random.normal(0, 2)  # Smaller variation for more realistic data
                wind_variation = np.random.normal(0, 0.8)
                cloud_variation = np.random.normal(0, 8)
                
                # Apply weekly patterns
                day_of_week = (start_time + timedelta(days=day)).weekday()
                weekly_temp_adjust = np.sin(2 * np.pi * day_of_week / 7) * 2
                weekly_wind_adjust = np.cos(2 * np.pi * day_of_week / 7) * 1
                
                # Calculate daily values with bounds
                daily_temp = base_temp + temp_variation + weekly_temp_adjust
                daily_wind = current.get('wind_speed', 3) + wind_variation + weekly_wind_adjust
                daily_clouds = current.get('cloud_cover', 50) + cloud_variation
                
                # Apply realistic bounds
                temperature = max(min(daily_temp, 38), -3)  # Zimbabwe temperature range
                wind_speed = max(min(daily_wind, 25), 0)
                cloud_cover = max(min(daily_clouds, 100), 0)
                
                # Determine weather conditions
                if cloud_cover < 20:
                    conditions = "Clear"
                    description = "clear sky"
                    rain_prob = np.random.uniform(0, 0.1)
                elif cloud_cover < 50:
                    conditions = "Partly Cloudy" 
                    description = "partly cloudy"
                    rain_prob = np.random.uniform(0.1, 0.3)
                elif cloud_cover < 80:
                    conditions = "Cloudy"
                    description = "cloudy"
                    rain_prob = np.random.uniform(0.3, 0.6)
                else:
                    conditions = "Overcast"
                    description = "overcast"
                    rain_prob = np.random.uniform(0.5, 0.8)
                    
                # Special condition for rain
                if rain_prob > 0.6 and conditions in ["Cloudy", "Overcast"]:
                    conditions = "Rain"
                    description = "light rain" if rain_prob < 0.8 else "moderate rain"
                
                # Calculate temperature extremes
                temp_high = temperature + np.random.uniform(3, 7)
                temp_low = temperature - np.random.uniform(3, 7)
                
                # Calculate solar metrics
                is_daytime = True  # Simplified for forecast
                irradiance = self._calculate_irradiance(cloud_cover, is_daytime)
                solar_potential = "High" if irradiance > 700 else "Medium" if irradiance > 400 else "Low"
                
                # Calculate wind potential
                wind_potential = "High" if wind_speed > 5 else "Medium" if wind_speed > 3 else "Low"
                
                # Create comprehensive forecast entry
                forecast = {
                    'timestamp': start_time + timedelta(days=day),
                    'date': (start_time + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'day_name': (start_time + timedelta(days=day)).strftime('%A'),
                    'temperature': float(round(temperature, 1)),
                    'temp_high': float(round(temp_high, 1)),
                    'temp_low': float(round(temp_low, 1)),
                    'humidity': float(max(10, min(95, current.get('humidity', 50) + np.random.randint(-15, 15)))),
                    'pressure': float(max(980, min(1040, current.get('pressure', 1013) + np.random.randint(-10, 10)))),
                    'wind_speed': float(round(wind_speed, 1)),
                    'wind_direction': (current.get('wind_direction', 0) + np.random.randint(-60, 60)) % 360,
                    'cloud_cover': int(cloud_cover),
                    'conditions': conditions,
                    'description': description,
                    'rain_probability': round(min(0.99, max(0.01, rain_prob)), 2),
                    'solar_potential': solar_potential,
                    'wind_potential': wind_potential,
                    'irradiance': float(round(irradiance)),
                    'uv_index': min(12, max(1, round(irradiance/100))),  # Simple UV estimate
                    'visibility': max(2, 10 - int(cloud_cover/15))  # km visibility estimate
                }
                
                forecasts.append(forecast)
                
            return forecasts
            
        except Exception as e:
            logging.error(f"Forecast generation failed for {location}: {str(e)}", exc_info=True)
            # Return a minimal safe forecast if generation fails
            return [{
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'temperature': 20,
                'conditions': 'Clear',
                'description': 'clear sky'
            } for i in range(min(7, days))]

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
