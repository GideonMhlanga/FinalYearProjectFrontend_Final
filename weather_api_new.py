import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WeatherAPI:
    """Enhanced weather API handler with robust error handling"""
    
    def __init__(self):
        self.base_url = "https://api.open-meteo.com/v1/forecast"
        self.timeout = 10
        
    def get_current_weather(self, location: str) -> Dict:
        """Get current weather with proper error handling"""
        try:
            params = {
                'latitude': location['lat'],
                'longitude': location['lon'],
                'current': 'temperature_2m,wind_speed_10m,relative_humidity_2m'
            }
            response = self._make_api_call(params)
            
            if not response or 'current' not in response:
                raise ValueError("Invalid API response structure")
                
            current = response['current']
            
            return {
                'temperature': current.get('temperature_2m', 0),
                'wind_speed': current.get('wind_speed_10m', 0),
                'humidity': current.get('relative_humidity_2m', 0),
                'conditions': self._get_weather_condition(current),
                'irradiance': self._calculate_irradiance(current),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Current weather error: {str(e)}")
            return self._get_fallback_current(location)

    def get_forecast(self, location: str) -> List[Dict]:
        """Get 7-day forecast with robust data validation"""
        try:
            params = {
                'latitude': location['lat'],
                'longitude': location['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,weather_code',
                'timezone': 'auto'
            }
            response = self._make_api_call(params)
            
            if not response or 'daily' not in response:
                raise ValueError("Invalid forecast response structure")
                
            return self._parse_forecast_data(response['daily'])
            
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return self._get_fallback_forecast(location)

    def _parse_forecast_data(self, daily_data: Dict) -> List[Dict]:
        """Safely parse forecast data with type checking"""
        forecast = []
        
        # Validate and prepare time data
        times = daily_data.get('time', [])
        if isinstance(times, str):
            times = [times]
        if not isinstance(times, list):
            times = []
            
        # Prepare other data arrays with safe defaults
        temp_max = self._ensure_list(daily_data.get('temperature_2m_max', []), len(times), 0)
        temp_min = self._ensure_list(daily_data.get('temperature_2m_min', []), len(times), 0)
        weather_codes = self._ensure_list(daily_data.get('weather_code', []), len(times), 0)
        
        for i in range(min(7, len(times))):  # Limit to 7 days
            forecast.append({
                'date': times[i],
                'day_name': (datetime.now() + timedelta(days=i)).strftime('%A'),
                'temperature': (temp_max[i] + temp_min[i]) / 2,
                'temp_high': temp_max[i],
                'temp_low': temp_min[i],
                'conditions': self._code_to_condition(weather_codes[i]),
                'irradiance': self._estimate_irradiance(weather_codes[i]),
                'wind_speed': self._estimate_wind(weather_codes[i]),
                'rain_probability': self._estimate_rain(weather_codes[i]),
                'solar_potential': self._get_solar_potential(weather_codes[i]),
                'wind_potential': self._get_wind_potential(weather_codes[i])
            })
            
        return forecast

    def _ensure_list(self, data, expected_length: int, default_value):
        """Ensure data is a list of expected length"""
        if isinstance(data, (int, float)):
            return [data] * expected_length
        if not isinstance(data, list):
            return [default_value] * expected_length
        if len(data) < expected_length:
            return data + [default_value] * (expected_length - len(data))
        return data[:expected_length]

    # Add all your helper methods (_calculate_irradiance, _code_to_condition, etc.)
    # ...

class AnomalyDetector:
    """Enhanced anomaly detector with timestamp handling"""
    
    def __init__(self):
        self.min_data_points = 24  # Minimum data points required for analysis
        
    def detect_anomalies(self, data: Union[pd.DataFrame, List[Dict]]) -> Dict:
        """Detect anomalies with robust timestamp handling"""
        try:
            df = self._prepare_data(data)
            
            if len(df) < self.min_data_points:
                raise ValueError(f"Insufficient data points ({len(df)} < {self.min_data_points})")
                
            anomalies = self._run_analysis(df)
            
            return {
                'summary': self._summarize_anomalies(anomalies),
                'anomalies': anomalies,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return self._empty_anomaly_result(str(e))

    def _prepare_data(self, data) -> pd.DataFrame:
        """Prepare DataFrame with proper timestamp handling"""
        if not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
            
        # Handle timestamp in different possible columns
        time_col = next(
            (col for col in ['timestamp', 'time', 'date', 'datetime', 'created_at'] 
             if col in df.columns), None)
            
        if time_col:
            df['timestamp'] = pd.to_datetime(df[time_col])
        else:
            # Generate synthetic timestamps if none exist
            freq = '15T' if len(df) > 48 else 'H'  # 15min or hourly
            df['timestamp'] = pd.date_range(
                end=pd.Timestamp.now(),
                periods=len(df),
                freq=freq
            )
            
        return df.set_index('timestamp').sort_index()

    def _run_analysis(self, df: pd.DataFrame) -> Dict:
        """Run actual anomaly detection algorithms"""
        # Implement your anomaly detection logic here
        # Example simple threshold-based detection:
        anomalies = {
            'solar': [],
            'wind': [],
            'battery': []
        }
        
        if 'solar_power' in df.columns:
            mean = df['solar_power'].mean()
            std = df['solar_power'].std()
            anomalies['solar'] = df[df['solar_power'] < (mean - 2*std)].to_dict('records')
            
        # Add similar checks for other metrics
        
        return anomalies

    def _summarize_anomalies(self, anomalies: Dict) -> Dict:
        """Generate anomaly summary statistics"""
        counts = {
            'total': 0,
            'severe': 0,
            'moderate': 0,
            'mild': 0
        }
        
        for category, items in anomalies.items():
            counts['total'] += len(items)
            # Classify severity based on your criteria
            counts['severe'] += sum(1 for item in items if item.get('severity', 0) > 0.8)
            counts['moderate'] += sum(1 for item in items if 0.5 < item.get('severity', 0) <= 0.8)
            counts['mild'] += sum(1 for item in items if item.get('severity', 0) <= 0.5)
            
        return counts

    def _empty_anomaly_result(self, error_msg: str = "") -> Dict:
        """Return empty anomaly result with error information"""
        return {
            'summary': {
                'total': 0,
                'severe': 0,
                'moderate': 0,
                'mild': 0,
                'error': error_msg
            },
            'anomalies': {},
            'timestamp': datetime.now().isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Initialize services
    weather_api = WeatherAPI()
    anomaly_detector = AnomalyDetector()
    
    # Example location
    location = {'lat': -20.15, 'lon': 28.58, 'name': 'Bulawayo'}
    
    # Get weather data
    current = weather_api.get_current_weather(location)
    forecast = weather_api.get_forecast(location)
    
    print("Current Weather:", current)
    print("Forecast:", forecast[:2])  # Print first 2 days
    
    # Generate some test data for anomaly detection
    test_data = [{
        'solar_power': np.random.normal(50, 10),
        'wind_power': np.random.normal(30, 5),
        'timestamp': (datetime.now() - timedelta(hours=i)).isoformat()
    } for i in range(72)]  # 72 hours of data
    
    # Detect anomalies
    anomalies = anomaly_detector.detect_anomalies(test_data)
    print("Anomaly Summary:", anomalies['summary'])