import os
import requests
import logging
from datetime import datetime
import pytz
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class WeatherAPI:
    """Weather API client for RapidAPI Weather API"""
    
    def __init__(self):
        self.api_key = os.getenv('RAPIDAPI_KEY', '7f9e16f0efmsh4bd796dccefbdc5p12f98ejsn46cba1899d09')
        self.base_url = "https://weather-api167.p.rapidapi.com/api/weather/forecast"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "weather-api167.p.rapidapi.com",
            "Accept": "application/json"
        }
        self.default_location = "Bulawayo"
        self.timezone = "Africa/Harare"
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # Available locations in Zimbabwe
        self.available_locations = [
            "Bulawayo", "Harare", "Gweru", "Mutare",
            "Victoria_Falls", "Chitungwiza", "Kwekwe",
            "Kadoma", "Masvingo", "Chinhoyi"
        ]

    def _make_api_request(self, params=None):
        """Make API request with caching"""
        cache_key = f"{str(params)}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if (datetime.now(pytz.utc) - cached_time).total_seconds() < self.cache_timeout:
                return cached_data
        
        try:
            # Make the API call
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Update cache
            self.cache[cache_key] = (datetime.now(pytz.utc), data)
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {str(e)}")
            raise Exception(f"Weather API request failed: {str(e)}")

    def _convert_kelvin_to_celsius(self, kelvin_temp):
        """Convert temperature from Kelvin to Celsius"""
        if kelvin_temp is None:
            return None
        return round(kelvin_temp - 273.15, 1)

    def get_current_weather(self, location=None):
        """Get current weather data"""
        location = location or self.default_location
        
        try:
            params = {
                "place": f"{location},ZW",
                "cnt": "1",
                "units": "metric",
                "type": "three_hour",
                "mode": "json",
                "lang": "en"
            }
            
            data = self._make_api_request(params)
            if not data.get('list'):
                raise Exception("No weather data returned from API")
                
            current = data['list'][0]
            main = current.get('main', {})
            weather = current.get('weather', [{}])[0]
            wind = current.get('wind', {})
            rain = current.get('rain', {})
            snow = current.get('snow', {})
            
            # Convert timestamp to local timezone
            dt = datetime.fromtimestamp(current['dt'], pytz.timezone(self.timezone))
            
            return {
                'timestamp': dt,
                'temperature': self._convert_kelvin_to_celsius(main.get('temprature')),
                'feels_like': self._convert_kelvin_to_celsius(main.get('temprature_feels_like')),
                'humidity': main.get('humidity'),
                'pressure': main.get('pressure'),
                'wind_speed': wind.get('speed'),
                'wind_dir': wind.get('dir', wind.get('degrees')),
                'wind_angle': wind.get('angle', wind.get('degrees')),
                'weather_icon': weather.get('icon'),
                'weather_summary': weather.get('description'),
                'weather_code': weather.get('id'),
                'cloud_cover': current.get('clouds', {}).get('cloudiness'),
                'precipitation': rain.get('amount', 0) or snow.get('amount', 0),
                'precipitation_type': 'rain' if rain else 'snow' if snow else 'none',
                'uv_index': None,  # Not available in this API
                'visibility': current.get('visibility_distance'),
                'location': location
            }
            
        except Exception as e:
            logging.error(f"Failed to get current weather: {str(e)}")
            raise Exception(f"Failed to get current weather: {str(e)}")

    def get_daily_forecast(self, location=None, days=5):
        """Get daily forecast data"""
        location = location or self.default_location
        
        try:
            params = {
                "place": f"{location},ZW",
                "cnt": str(days * 8),  # 8 readings per day (3-hour intervals)
                "units": "metric",
                "type": "three_hour",
                "mode": "json",
                "lang": "en"
            }
            
            data = self._make_api_request(params)
            if not data.get('list'):
                raise Exception("No weather data returned from API")
                
            # Group by day
            daily_data = {}
            for forecast in data['list']:
                date = datetime.fromtimestamp(forecast['dt']).strftime('%Y-%m-%d')
                if date not in daily_data:
                    daily_data[date] = []
                daily_data[date].append(forecast)
            
            forecasts = []
            for date, day_forecasts in list(daily_data.items())[:days]:
                # Calculate day stats
                temps = [f['main']['temprature'] for f in day_forecasts]
                temp_max = max(temps)
                temp_min = min(temps)
                
                # Use midday forecast for day summary
                midday_forecast = day_forecasts[len(day_forecasts)//2]
                main = midday_forecast.get('main', {})
                weather = midday_forecast.get('weather', [{}])[0]
                wind = midday_forecast.get('wind', {})
                
                forecast_date = datetime.strptime(date, '%Y-%m-%d')
                forecast_date = forecast_date.replace(tzinfo=pytz.timezone(self.timezone))
                
                forecasts.append({
                    'date': date,
                    'day_name': forecast_date.strftime('%A'),
                    'weather_icon': weather.get('icon'),
                    'weather_summary': weather.get('description'),
                    'weather_code': weather.get('id'),
                    'temperature': self._convert_kelvin_to_celsius(main.get('temprature')),
                    'temp_max': self._convert_kelvin_to_celsius(temp_max),
                    'temp_min': self._convert_kelvin_to_celsius(temp_min),
                    'humidity': main.get('humidity'),
                    'pressure': main.get('pressure'),
                    'wind_speed': wind.get('speed'),
                    'wind_dir': wind.get('dir', wind.get('degrees')),
                    'wind_angle': wind.get('angle', wind.get('degrees')),
                    'cloud_cover': midday_forecast.get('clouds', {}).get('cloudiness'),
                    'precipitation': sum(f.get('rain', {}).get('amount', 0) or 
                                       f.get('snow', {}).get('amount', 0) for f in day_forecasts),
                    'precipitation_type': 'rain' if any(f.get('rain') for f in day_forecasts) else 
                                         'snow' if any(f.get('snow') for f in day_forecasts) else 'none',
                    'uv_index': None,  # Not available in this API
                    'sunrise': None,  # Not available in hourly data
                    'sunset': None,   # Not available in hourly data
                    'location': location
                })
            
            return forecasts
            
        except Exception as e:
            logging.error(f"Failed to get daily forecast: {str(e)}")
            raise Exception(f"Failed to get daily forecast: {str(e)}")

    def get_hourly_forecast(self, location=None, hours=24):
        """Get hourly forecast data for today"""
        location = location or self.default_location
        
        try:
            params = {
                "place": f"{location},ZW",
                "cnt": str(hours // 3),  # 3-hour intervals
                "units": "metric",
                "type": "three_hour",
                "mode": "json",
                "lang": "en"
            }
            
            data = self._make_api_request(params)
            if not data.get('list'):
                raise Exception("No weather data returned from API")
                
            forecasts = []
            for forecast in data['list'][:hours]:
                main = forecast.get('main', {})
                weather = forecast.get('weather', [{}])[0]
                wind = forecast.get('wind', {})
                rain = forecast.get('rain', {})
                snow = forecast.get('snow', {})
                
                forecast_time = datetime.fromtimestamp(forecast['dt'], pytz.timezone(self.timezone))
                
                forecasts.append({
                    'time': forecast_time.strftime('%H:%M'),
                    'date': forecast_time.strftime('%Y-%m-%d'),
                    'weather_icon': weather.get('icon'),
                    'weather_summary': weather.get('description'),
                    'weather_code': weather.get('id'),
                    'temperature': self._convert_kelvin_to_celsius(main.get('temprature')),
                    'feels_like': self._convert_kelvin_to_celsius(main.get('temprature_feels_like')),
                    'humidity': main.get('humidity'),
                    'pressure': main.get('pressure'),
                    'wind_speed': wind.get('speed'),
                    'wind_dir': wind.get('dir', wind.get('degrees')),
                    'wind_angle': wind.get('angle', wind.get('degrees')),
                    'cloud_cover': forecast.get('clouds', {}).get('cloudiness'),
                    'precipitation': rain.get('amount', 0) or snow.get('amount', 0),
                    'precipitation_type': 'rain' if rain else 'snow' if snow else 'none',
                    'uv_index': None,  # Not available in this API
                    'visibility': forecast.get('visibility_distance'),
                    'location': location
                })
            
            return forecasts
            
        except Exception as e:
            logging.error(f"Failed to get hourly forecast: {str(e)}")
            raise Exception(f"Failed to get hourly forecast: {str(e)}")

    def get_combined_forecast(self, location=None, days=5, hours=24):
        """Get combined daily and hourly forecast"""
        try:
            return {
                'current': self.get_current_weather(location),
                'daily': self.get_daily_forecast(location, days),
                'hourly': self.get_hourly_forecast(location, hours)
            }
        except Exception as e:
            logging.error(f"Failed to get combined forecast: {str(e)}")
            raise Exception(f"Failed to get combined forecast: {str(e)}")

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

# Display status
if not is_api_working:
    st.error("⚠️ Weather API is currently unavailable. Please try again later.")