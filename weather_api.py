import os
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pytz
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

@dataclass
class CurrentWeather:
    """Dataclass for current weather conditions"""
    temperature: float  # in °C
    feels_like: float  # in °C
    humidity: int  # percentage
    wind_speed: float  # in m/s
    wind_direction: Optional[int]  # degrees
    cloud_cover: int  # percentage
    conditions: str  # e.g. "Clear", "Rain"
    description: str  # e.g. "light rain"
    irradiance: float  # estimated solar irradiance in W/m²
    is_day: bool  # whether it's daytime
    precipitation: float  # in mm
    pressure: int  # in hPa
    visibility: int  # in meters
    sunrise: datetime
    sunset: datetime
    location: str

@dataclass
class ForecastDay:
    """Dataclass for daily forecast data"""
    date: str  # YYYY-MM-DD
    day_name: str  # e.g. "Monday"
    temp_high: float  # in °C
    temp_low: float  # in °C
    temp_avg: float  # in °C
    wind_speed: float  # in m/s
    wind_direction: Optional[int]  # degrees
    cloud_cover: int  # percentage
    conditions: str  # e.g. "Clear"
    description: str  # e.g. "light rain"
    solar_potential: str  # "High", "Medium", or "Low"
    wind_potential: str  # "High", "Medium", or "Low"
    precipitation_prob: float  # percentage
    precipitation_amount: float  # in mm
    sunrise: datetime
    sunset: datetime

class WeatherAPI:
    """Weather API client for RapidAPI's OpenWeather endpoint"""
    
    def __init__(self):
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.host = os.getenv("RAPIDAPI_HOST", "open-weather13.p.rapidapi.com")
        self.base_url = f"https://{self.host}"
        
        if not self.api_key:
            raise ValueError("Missing RapidAPI key. Set RAPIDAPI_KEY in .env file")
        
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-host": self.host,
            "x-rapidapi-key": self.api_key
        })
        
        # Zimbabwe timezone
        self.tz = pytz.timezone("Africa/Harare")
        
        # Zimbabwe cities with coordinates - Bulawayo as default
        self.locations = {
            "Bulawayo": {"lat": -20.1325, "lon": 28.6265},
            "Harare": {"lat": -17.8252, "lon": 31.0335},
            "Mutare": {"lat": -18.9726, "lon": 32.6636},
            "Gweru": {"lat": -19.4566, "lon": 29.8025},
            "Kwekwe": {"lat": -18.9286, "lon": 29.8159},
            "Chitungwiza": {"lat": -18.0128, "lon": 31.0756},
            "Masvingo": {"lat": -20.0744, "lon": 30.8328},
            "Chinhoyi": {"lat": -17.3667, "lon": 30.2000},
            "Marondera": {"lat": -18.1833, "lon": 31.5500},
            "Bindura": {"lat": -17.3000, "lon": 31.3333},
            "Victoria Falls": {"lat": -17.9243, "lon": 25.8572},
            "Kariba": {"lat": -16.5167, "lon": 28.8000}
        }
        
        # Set Bulawayo as default location
        self.default_location = "Bulawayo"
        self.default_lat = self.locations["Bulawayo"]["lat"]
        self.default_lon = self.locations["Bulawayo"]["lon"]
    
    def get_available_locations(self) -> List[str]:
        """Get list of available locations in Zimbabwe"""
        return sorted(self.locations.keys())
    
    def get_current_weather(self, location: str = None) -> CurrentWeather:
        """
        Get current weather conditions for specified location or default location
        
        Args:
            location: Optional location name. Uses Bulawayo as default if None
            
        Returns:
            CurrentWeather object with all current conditions
        """
        if location is None:
            location = self.default_location
        
        if location not in self.locations:
            raise ValueError(f"Location '{location}' not available. Choose from: {', '.join(self.get_available_locations())}")
            
        lat, lon = self.locations[location]["lat"], self.locations[location]["lon"]
        
        try:
            # Get current weather
            current_url = f"{self.base_url}/city/{lat}/{lon}"
            response = self._make_api_request(current_url)
            
            # Process response
            weather = response["weather"][0]
            main = response["main"]
            wind = response["wind"]
            clouds = response["clouds"]
            sys = response["sys"]
            
            # Calculate solar irradiance
            now = datetime.now(self.tz)
            sunrise = datetime.fromtimestamp(sys["sunrise"], tz=self.tz)
            sunset = datetime.fromtimestamp(sys["sunset"], tz=self.tz)
            is_day = sunrise < now < sunset
            irradiance = self._calculate_irradiance(clouds["all"], is_day)
            
            return CurrentWeather(
                temperature=main["temp"] - 273.15,  # Convert K to °C
                feels_like=main["feels_like"] - 273.15,
                humidity=main["humidity"],
                wind_speed=wind["speed"],
                wind_direction=wind.get("deg"),
                cloud_cover=clouds["all"],
                conditions=weather["main"],
                description=weather["description"],
                irradiance=irradiance,
                is_day=is_day,
                precipitation=response.get("rain", {}).get("1h", 0),
                pressure=main["pressure"],
                visibility=response.get("visibility", 10000),
                sunrise=sunrise,
                sunset=sunset,
                location=location
            )
            
        except Exception as e:
            raise Exception(f"Failed to get current weather for {location}: {str(e)}")

    def get_forecast(self, days: int = 5, location: str = None) -> List[ForecastDay]:
        """
        Get weather forecast for specified number of days
        
        Args:
            days: Number of forecast days (1-5)
            location: Optional location name. Uses Bulawayo as default if None
            
        Returns:
            List of ForecastDay objects
        """
        if not 1 <= days <= 5:
            raise ValueError("Forecast can only be retrieved for 1-5 days")
            
        if location is None:
            location = self.default_location
            
        if location not in self.locations:
            raise ValueError(f"Location '{location}' not available")
            
        lat, lon = self.locations[location]["lat"], self.locations[location]["lon"]
        
        try:
            # Get 5-day forecast
            forecast_url = f"{self.base_url}/city/fivedaysforcast/{lat}/{lon}"
            forecast_data = self._make_api_request(forecast_url)
            
            # Get city info for sunrise/sunset
            city_url = f"{self.base_url}/city/{lat}/{lon}"
            city_data = self._make_api_request(city_url)
            base_sunrise = datetime.fromtimestamp(city_data["sys"]["sunrise"], tz=self.tz)
            base_sunset = datetime.fromtimestamp(city_data["sys"]["sunset"], tz=self.tz)
            
            return self._process_forecast(forecast_data, days, base_sunrise, base_sunset, location)
            
        except Exception as e:
            raise Exception(f"Failed to get forecast for {location}: {str(e)}")

    def _make_api_request(self, url: str) -> Dict:
        """Make API request with proper error handling"""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 401:
                raise Exception("Invalid RapidAPI key - please check your API key")
            elif response.status_code == 429:
                raise Exception("API rate limit exceeded")
            else:
                raise Exception(f"HTTP error: {http_err}")
        except Exception as e:
            raise Exception(f"API request failed: {str(e)}")

    def _calculate_irradiance(self, cloud_cover: int, is_day: bool) -> float:
        """Estimate solar irradiance based on cloud cover and time of day"""
        if not is_day:
            return 0.0
        
        # Clear sky irradiance (W/m²) adjusted for cloud cover
        clear_sky = 1000  # Maximum expected irradiance
        return max(0, clear_sky * (1 - cloud_cover / 100))

    def _process_forecast(self, forecast_data: Dict, days: int, 
                         base_sunrise: datetime, base_sunset: datetime,
                         location: str) -> List[ForecastDay]:
        """Process raw forecast data into ForecastDay objects"""
        daily_forecasts = []
        
        # Group 3-hour forecasts by day
        daily_data = {}
        for item in forecast_data.get("list", []):
            date_str = item.get("dt_txt", "")[:10]  # Extract YYYY-MM-DD
            if not date_str:
                continue
                
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(item)
        
        # Calculate sunrise/sunset for each day
        sun_times = {}
        for i in range(days):
            date = (datetime.now(self.tz) + timedelta(days=i)).date()
            date_str = date.strftime("%Y-%m-%d")
            sun_times[date_str] = (
                base_sunrise + timedelta(days=i),
                base_sunset + timedelta(days=i)
            )
        
        # Process each day
        for date, day_items in list(daily_data.items())[:days]:
            date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            sunrise, sunset = sun_times.get(date, (None, None))
            
            # Calculate aggregates
            temps = [item["main"]["temp"] - 273.15 for item in day_items]
            wind_speeds = [item.get("wind", {}).get("speed", 0) for item in day_items]
            wind_dirs = [item.get("wind", {}).get("deg") for item in day_items]
            cloud_covers = [item.get("clouds", {}).get("all", 0) for item in day_items]
            pops = [item.get("pop", 0) for item in day_items]  # Probability of precipitation
            
            # Precipitation amount
            precip_amounts = [
                item.get("rain", {}).get("3h", 0) or 
                item.get("snow", {}).get("3h", 0)
                for item in day_items
            ]
            
            # Most common weather condition
            conditions = [item["weather"][0]["main"] for item in day_items]
            most_common_condition = max(set(conditions), key=conditions.count)
            
            # Determine daylight hours for solar potential
            daylight_hours = [item for item in day_items 
                             if sunrise and sunset and
                             sunrise.time() <= datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S").time() <= sunset.time()]
            
            # Calculate solar potential
            avg_cloud_cover = sum(cloud_covers) / len(cloud_covers)
            has_daylight = len(daylight_hours) > 0
            
            if avg_cloud_cover < 30 and has_daylight:
                solar_potential = "High"
            elif avg_cloud_cover < 70 and has_daylight:
                solar_potential = "Medium"
            else:
                solar_potential = "Low"
            
            # Determine wind potential
            avg_wind_speed = sum(wind_speeds) / len(wind_speeds)
            if avg_wind_speed > 5:
                wind_potential = "High"
            elif avg_wind_speed > 3:
                wind_potential = "Medium"
            else:
                wind_potential = "Low"
            
            # Average wind direction (circular mean)
            if any(wind_dirs):
                avg_wind_dir = self._circular_mean([d for d in wind_dirs if d is not None])
            else:
                avg_wind_dir = None
            
            daily_forecasts.append(ForecastDay(
                date=date,
                day_name=date_obj.strftime("%A"),
                temp_high=max(temps),
                temp_low=min(temps),
                temp_avg=sum(temps) / len(temps),
                wind_speed=avg_wind_speed,
                wind_direction=avg_wind_dir,
                cloud_cover=int(avg_cloud_cover),
                conditions=most_common_condition,
                description=day_items[0]["weather"][0]["description"],
                solar_potential=solar_potential,
                wind_potential=wind_potential,
                precipitation_prob=max(pops) * 100,
                precipitation_amount=sum(precip_amounts),
                sunrise=sunrise,
                sunset=sunset
            ))
        
        return daily_forecasts

    def _circular_mean(self, angles: List[int]) -> int:
        """Calculate circular mean of wind directions"""
        if not angles:
            return None
            
        sin_sum = sum([np.sin(np.deg2rad(a)) for a in angles])
        cos_sum = sum([np.cos(np.deg2rad(a)) for a in angles])
        return int(np.rad2deg(np.arctan2(sin_sum, cos_sum))) % 360

# Create singleton instance for import
weather_api = WeatherAPI()