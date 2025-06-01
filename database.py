import os
import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from contextlib import contextmanager

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import and_

# Get database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gmhlanga:gmhlanga.2001@localhost:5432/frontend_finalproject")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    """User account information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    email = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    address = Column(String(200), nullable=True)
    role = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.now)
    last_login = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

class SystemSetting(Base):
    """System settings and configuration"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True)
    setting_name = Column(String(50), unique=True, nullable=False)
    setting_value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    
    def to_dict(self):
        try:
            value = json.loads(self.setting_value)
        except:
            value = self.setting_value
        return {
            "id": self.id,
            "setting_name": self.setting_name,
            "setting_value": value,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class BatterySpecs(Base):
    """Battery specifications and configuration"""
    __tablename__ = "battery_specs"
    
    id = Column(Integer, primary_key=True)
    battery_type = Column(String(100), nullable=False)
    capacity = Column(String(20), nullable=False)
    nominal_voltage = Column(String(20), nullable=False)
    max_charging_rate = Column(String(20), nullable=False)
    expected_lifespan = Column(String(20), nullable=False)
    installation_date = Column(String(20), nullable=False)
    updated_at = Column(DateTime, default=datetime.datetime.now, onupdate=datetime.datetime.now)
    
    def to_dict(self):
        return {
            "Type": self.battery_type,
            "Capacity": self.capacity,
            "Nominal Voltage": self.nominal_voltage,
            "Max Charging Rate": self.max_charging_rate,
            "Expected Lifespan": self.expected_lifespan,
            "Installation Date": self.installation_date
        }

class PowerData(Base):
    """Historical power generation and consumption data"""
    __tablename__ = "power_data"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    solar_power = Column(Float)
    wind_power = Column(Float)
    load = Column(Float)
    battery_soc = Column(Float)
    battery_voltage = Column(Float)
    battery_current = Column(Float)
    battery_temperature = Column(Float)
    irradiance = Column(Float)
    wind_speed = Column(Float)
    temperature = Column(Float)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "solar_power": self.solar_power,
            "wind_power": self.wind_power,
            "load": self.load,
            "battery_soc": self.battery_soc,
            "battery_voltage": self.battery_voltage,
            "battery_current": self.battery_current,
            "battery_temperature": self.battery_temperature,
            "irradiance": self.irradiance,
            "wind_speed": self.wind_speed,
            "temperature": self.temperature
        }

class SystemLog(Base):
    """System event and alert logs"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, index=True)
    log_type = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text)
    
    def to_dict(self):
        try:
            details = json.loads(self.details) if self.details else None
        except:
            details = self.details
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "log_type": self.log_type,
            "message": self.message,
            "details": details
        }

class WeatherData(Base):
    """Historical weather data"""
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    location = Column(String(50), nullable=False)
    temperature = Column(Float)
    condition = Column(String(50))
    wind_speed = Column(Float)
    wind_direction = Column(String(10))
    humidity = Column(Float)
    pressure = Column(Float)
    irradiance = Column(Float)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "location": self.location,
            "temperature": self.temperature,
            "condition": self.condition,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "humidity": self.humidity,
            "pressure": self.pressure,
            "irradiance": self.irradiance
        }

class BlockchainLog(Base):
    """Blockchain energy data logs"""
    __tablename__ = "blockchain_logs"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, index=True)
    data_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    data_hash = Column(String(128), nullable=False)
    transaction_hash = Column(String(128), nullable=False)
    blockchain_network = Column(String(50), default="Simulation")
    status = Column(String(20), default="confirmed")
    data_json = Column(Text, nullable=True)
    
    def to_dict(self):
        try:
            data = json.loads(self.data_json) if self.data_json else None
        except:
            data = self.data_json
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data_type": self.data_type,
            "description": self.description,
            "data_hash": self.data_hash,
            "transaction_hash": self.transaction_hash,
            "blockchain_network": self.blockchain_network,
            "status": self.status,
            "data": data
        }

class PredictiveMaintenance(Base):
    """Predictive maintenance data and forecasts"""
    __tablename__ = "predictive_maintenance"
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.now, index=True)
    component = Column(String(50), nullable=False)
    health_score = Column(Float, nullable=False)
    predicted_failure_date = Column(DateTime, nullable=True)
    maintenance_recommended = Column(Boolean, default=False)
    recommendation = Column(Text, nullable=True)
    confidence = Column(Float)
    analysis_data = Column(Text, nullable=True)
    maintenance_cost = Column(Float, nullable=True)
    failure_cost = Column(Float, nullable=True)
    
    def to_dict(self):
        try:
            analysis = json.loads(self.analysis_data) if self.analysis_data else None
        except:
            analysis = self.analysis_data
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "component": self.component,
            "health_score": self.health_score,
            "predicted_failure_date": self.predicted_failure_date.isoformat() if self.predicted_failure_date else None,
            "maintenance_recommended": self.maintenance_recommended,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "analysis_data": analysis,
            "maintenance_cost": self.maintenance_cost,
            "failure_cost": self.failure_cost
        }

class DatabaseManager:
    """Manager for database operations"""
    
    def __init__(self):
        self.engine = engine
        self.Session = Session
        Base.metadata.create_all(self.engine)
        self._seed_initial_data()
    
    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.add_system_log("error", "Database operation failed", {"error": str(e)})
            raise
        finally:
            session.close()
    
    def _seed_initial_data(self):
        """Seed initial data if database is empty"""
        with self.session_scope() as session:
            # Users
            if session.query(User).count() == 0:
                admin_user = User(
                    username="admin",
                    password="admin123",
                    role="admin"
                )
                session.add(admin_user)
            
            # System Settings
            if session.query(SystemSetting).count() == 0:
                settings = [
                    SystemSetting(
                        setting_name="battery",
                        setting_value=json.dumps({
                            "capacity": 10.0,
                            "charge_rate": 2.0,
                            "discharge_rate": 2.0,
                            "min_soc": 20.0,
                            "max_soc": 90.0,
                            "efficiency": 92.0,
                            "chemistry": "Lithium Ion"
                        })
                    ),
                    SystemSetting(
                        setting_name="system",
                        setting_value=json.dumps({
                            "solar_capacity": 5.0,
                            "wind_capacity": 3.0,
                            "backup_enabled": True,
                            "backup_threshold": 30.0,
                            "grid_connected": False
                        })
                    ),
                    SystemSetting(
                        setting_name="location",
                        setting_value=json.dumps({
                            "name": "Harare",
                            "latitude": -17.8292,
                            "longitude": 31.0522,
                            "timezone": "Africa/Harare",
                            "country": "Zimbabwe"
                        })
                    )
                ]
                session.add_all(settings)
            
            # Battery Specs
            if session.query(BatterySpecs).count() == 0:
                battery_specs = BatterySpecs(
                    battery_type="Lithium Iron Phosphate (LiFePO4)",
                    capacity="20 kWh",
                    nominal_voltage="48V",
                    max_charging_rate="5 kW",
                    expected_lifespan="~3000 cycles",
                    installation_date="2023-01-15"
                )
                session.add(battery_specs)
            
            # Initial log
            if session.query(SystemLog).count() == 0:
                initial_log = SystemLog(
                    log_type="info",
                    message="System initialized",
                    details=json.dumps({"source": "DatabaseManager"})
                )
                session.add(initial_log)

    # User Management Methods
    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users (excluding passwords)"""
        with self.session_scope() as session:
            return [user.to_dict() for user in session.query(User).all()]
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """Authenticate user credentials"""
        with self.session_scope() as session:
            user = session.query(User).filter_by(username=username).first()
            if user and user.password == password:
                return {"authenticated": True, "role": user.role, "user": user.to_dict()}
            return {"authenticated": False, "role": None, "user": None}
    
    def add_user(self, username: str, password: str, role: str) -> bool:
        """Add a new user"""
        try:
            with self.session_scope() as session:
                if session.query(User).filter_by(username=username).first():
                    return False
                user = User(username=username, password=password, role=role)
                session.add(user)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to add user", {"username": username, "error": str(e)})
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete a user (except admin)"""
        if username == "admin":
            return False
            
        try:
            with self.session_scope() as session:
                user = session.query(User).filter_by(username=username).first()
                if user:
                    session.delete(user)
                    return True
                return False
        except Exception as e:
            self.add_system_log("error", "Failed to delete user", {"username": username, "error": str(e)})
            return False
    
    def update_user_profile(self, username: str, profile_data: Dict[str, Any]) -> bool:
        """Update user profile information"""
        try:
            with self.session_scope() as session:
                user = session.query(User).filter_by(username=username).first()
                if not user:
                    return False
                
                if "first_name" in profile_data:
                    user.first_name = profile_data["first_name"]
                if "last_name" in profile_data:
                    user.last_name = profile_data["last_name"]
                if "email" in profile_data:
                    user.email = profile_data["email"]
                if "phone" in profile_data:
                    user.phone = profile_data["phone"]
                if "address" in profile_data:
                    user.address = profile_data["address"]
                if "role" in profile_data and username != "admin":
                    user.role = profile_data["role"]
                
                return True
        except Exception as e:
            self.add_system_log("error", "Failed to update user profile", {"username": username, "error": str(e)})
            return False
    
    def change_user_password(self, username: str, new_password: str) -> bool:
        """Change a user's password"""
        try:
            with self.session_scope() as session:
                user = session.query(User).filter_by(username=username).first()
                if user:
                    user.password = new_password
                    return True
                return False
        except Exception as e:
            self.add_system_log("error", "Failed to change password", {"username": username, "error": str(e)})
            return False
    
    def record_user_login(self, username: str) -> bool:
        """Update the last_login timestamp for a user"""
        try:
            with self.session_scope() as session:
                user = session.query(User).filter_by(username=username).first()
                if user:
                    user.last_login = datetime.datetime.now()
                    return True
                return False
        except Exception as e:
            self.add_system_log("error", "Failed to record login", {"username": username, "error": str(e)})
            return False

    # System Settings Methods
    def get_settings(self, setting_name: Optional[str] = None) -> Dict[str, Any]:
        """Get system settings"""
        with self.session_scope() as session:
            if setting_name:
                setting = session.query(SystemSetting).filter_by(setting_name=setting_name).first()
                return setting.to_dict()["setting_value"] if setting else {}
            else:
                settings = session.query(SystemSetting).all()
                return {setting.setting_name: setting.to_dict()["setting_value"] for setting in settings}
    
    def update_settings(self, setting_name: str, setting_value: Union[Dict[str, Any], str]) -> bool:
        """Update system settings"""
        try:
            setting_value_str = json.dumps(setting_value) if isinstance(setting_value, dict) else setting_value
            
            with self.session_scope() as session:
                setting = session.query(SystemSetting).filter_by(setting_name=setting_name).first()
                if setting:
                    setting.setting_value = setting_value_str
                else:
                    setting = SystemSetting(setting_name=setting_name, setting_value=setting_value_str)
                    session.add(setting)
                return True
        except Exception as e:
            self.add_system_log("error", "Failed to update settings", {"setting": setting_name, "error": str(e)})
            return False

    # Battery Specs Methods
    def get_battery_specs(self) -> Dict[str, str]:
        """Get the most recent battery specifications"""
        with self.session_scope() as session:
            specs = session.query(BatterySpecs).order_by(BatterySpecs.updated_at.desc()).first()
            return specs.to_dict() if specs else {
                "Type": "Lithium Iron Phosphate (LiFePO4)",
                "Capacity": "20 kWh",
                "Nominal Voltage": "48V",
                "Max Charging Rate": "5 kW",
                "Expected Lifespan": "~3000 cycles",
                "Installation Date": "2023-01-15"
            }
    
    def save_battery_specs(self, specs: Dict[str, str]) -> bool:
        """Save new battery specifications"""
        try:
            with self.session_scope() as session:
                battery_specs = BatterySpecs(
                    battery_type=specs["Type"],
                    capacity=specs["Capacity"],
                    nominal_voltage=specs["Nominal Voltage"],
                    max_charging_rate=specs["Max Charging Rate"],
                    expected_lifespan=specs["Expected Lifespan"],
                    installation_date=specs["Installation Date"]
                )
                session.add(battery_specs)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to save battery specs", {"error": str(e)})
            return False

    # Power Data Methods
    def save_power_data(self, data: Dict[str, Any]) -> bool:
        """Save power generation/consumption data"""
        try:
            with self.session_scope() as session:
                power_data = PowerData(
                    timestamp=data.get("timestamp", datetime.datetime.now()),
                    solar_power=data.get("solar_power"),
                    wind_power=data.get("wind_power"),
                    load=data.get("load"),
                    battery_soc=data.get("battery_soc"),
                    battery_voltage=data.get("battery_voltage"),
                    battery_current=data.get("battery_current"),
                    battery_temperature=data.get("battery_temperature"),
                    irradiance=data.get("irradiance"),
                    wind_speed=data.get("wind_speed"),
                    temperature=data.get("temperature")
                )
                session.add(power_data)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to save power data", {"error": str(e)})
            return False
    
    def get_power_data(self, timeframe: str = "day") -> pd.DataFrame:
        """Get historical power data"""
        with self.session_scope() as session:
            now = datetime.datetime.now()
            if timeframe == "day":
                start_time = now - datetime.timedelta(days=1)
            elif timeframe == "week":
                start_time = now - datetime.timedelta(weeks=1)
            elif timeframe == "month":
                start_time = now - datetime.timedelta(days=30)
            else:
                start_time = now - datetime.timedelta(days=1)
            
            data = session.query(PowerData).filter(PowerData.timestamp >= start_time).order_by(PowerData.timestamp).all()
            
            if data:
                df = pd.DataFrame([d.to_dict() for d in data])
                df["total_generation"] = df["solar_power"] + df["wind_power"]
                df["net_power"] = df["total_generation"] - df["load"]
            else:
                df = pd.DataFrame(columns=[
                    "timestamp", "solar_power", "wind_power", "load", "battery_soc",
                    "battery_voltage", "battery_current", "battery_temperature",
                    "irradiance", "wind_speed", "temperature", "total_generation", "net_power"
                ])
            return df
    
    def get_latest_power_data(self) -> Dict[str, Any]:
        """Get the most recent power data entry"""
        with self.session_scope() as session:
            data = session.query(PowerData).order_by(PowerData.timestamp.desc()).first()
            return data.to_dict() if data else {}
    
    def get_power_stats(self, timeframe: str = "day") -> Dict[str, Any]:
        """Get statistics for power data in the given timeframe"""
        df = self.get_power_data(timeframe)
        if df.empty:
            return {}
        
        return {
            "total_solar": df["solar_power"].sum(),
            "total_wind": df["wind_power"].sum(),
            "total_load": df["load"].sum(),
            "avg_battery_soc": df["battery_soc"].mean(),
            "max_battery_temp": df["battery_temperature"].max(),
            "min_battery_temp": df["battery_temperature"].min(),
            "energy_balance": df["total_generation"].sum() - df["load"].sum()
        }

    # Weather Data Methods
    def save_weather_data(self, data: Dict[str, Any]) -> bool:
        """Save weather data"""
        try:
            with self.session_scope() as session:
                weather_data = WeatherData(
                    timestamp=data.get("timestamp", datetime.datetime.now()),
                    location=data.get("location", "Harare"),
                    temperature=data.get("temperature"),
                    condition=data.get("condition"),
                    wind_speed=data.get("wind_speed"),
                    wind_direction=data.get("wind_direction"),
                    humidity=data.get("humidity"),
                    pressure=data.get("pressure"),
                    irradiance=data.get("irradiance")
                )
                session.add(weather_data)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to save weather data", {"error": str(e)})
            return False
    
    def get_latest_weather_data(self) -> Dict[str, Any]:
        """Get the most recent weather data entry"""
        with self.session_scope() as session:
            data = session.query(WeatherData).order_by(WeatherData.timestamp.desc()).first()
            return data.to_dict() if data else {}
    
    def get_weather_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get weather statistics for the past N days"""
        with self.session_scope() as session:
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            data = session.query(WeatherData).filter(WeatherData.timestamp >= cutoff).all()
            
            if not data:
                return {}
            
            df = pd.DataFrame([d.to_dict() for d in data])
            return {
                "avg_temperature": df["temperature"].mean(),
                "max_temperature": df["temperature"].max(),
                "min_temperature": df["temperature"].min(),
                "avg_wind_speed": df["wind_speed"].mean(),
                "max_wind_speed": df["wind_speed"].max(),
                "total_irradiance": df["irradiance"].sum(),
                "most_common_condition": df["condition"].mode()[0] if not df["condition"].mode().empty else "Unknown"
            }

    # System Log Methods
    def add_system_log(self, log_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """Add a system log entry"""
        try:
            details_str = json.dumps(details) if details else None
            with self.session_scope() as session:
                log = SystemLog(log_type=log_type, message=message, details=details_str)
                session.add(log)
            return True
        except Exception as e:
            print(f"Failed to add system log: {e}")
            return False
    
    def get_system_logs(self, limit: int = 50, log_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get system logs"""
        with self.session_scope() as session:
            query = session.query(SystemLog).order_by(SystemLog.timestamp.desc())
            if log_type:
                query = query.filter(SystemLog.log_type == log_type)
            return [log.to_dict() for log in query.limit(limit).all()]
    
    def clear_old_logs(self, days_to_keep: int = 30) -> int:
        """Clear logs older than specified days"""
        with self.session_scope() as session:
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            deleted_count = session.query(SystemLog).filter(SystemLog.timestamp < cutoff).delete()
            return deleted_count

    # Blockchain Methods
    def save_blockchain_log(self, data_type: str, data_hash: str, transaction_hash: str, 
                          description: str = "", data_json: Optional[str] = None,
                          blockchain_network: str = "Simulation", status: str = "confirmed") -> bool:
        """Save a blockchain log entry"""
        try:
            with self.session_scope() as session:
                blockchain_log = BlockchainLog(
                    timestamp=datetime.datetime.now(),
                    data_type=data_type,
                    description=description,
                    data_hash=data_hash,
                    transaction_hash=transaction_hash,
                    blockchain_network=blockchain_network,
                    status=status,
                    data_json=data_json
                )
                session.add(blockchain_log)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to save blockchain log", {"error": str(e)})
            return False
    
    def get_blockchain_logs(self, limit: int = 20, data_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get blockchain logs"""
        with self.session_scope() as session:
            query = session.query(BlockchainLog).order_by(BlockchainLog.timestamp.desc())
            if data_type:
                query = query.filter(BlockchainLog.data_type == data_type)
            return [log.to_dict() for log in query.limit(limit).all()]
    
    def get_blockchain_log_by_transaction(self, transaction_hash: str) -> Optional[Dict[str, Any]]:
        """Get a blockchain log by transaction hash"""
        with self.session_scope() as session:
            log = session.query(BlockchainLog).filter_by(transaction_hash=transaction_hash).first()
            return log.to_dict() if log else None
    
    def get_blockchain_statistics(self) -> Dict[str, Any]:
        """Get statistics about blockchain logs"""
        with self.session_scope() as session:
            total_logs = session.query(func.count(BlockchainLog.id)).scalar()
            
            type_counts = {}
            for data_type, count in session.query(BlockchainLog.data_type, func.count(BlockchainLog.id)).\
                                         group_by(BlockchainLog.data_type).all():
                type_counts[data_type] = count
            
            network_counts = {}
            for network, count in session.query(BlockchainLog.blockchain_network, func.count(BlockchainLog.id)).\
                                       group_by(BlockchainLog.blockchain_network).all():
                network_counts[network] = count
            
            latest_log = session.query(BlockchainLog).order_by(BlockchainLog.timestamp.desc()).first()
            
            return {
                "total_logs": total_logs,
                "by_data_type": type_counts,
                "by_network": network_counts,
                "latest_log": latest_log.to_dict() if latest_log else None
            }

    # Predictive Maintenance Methods
    def save_predictive_maintenance(self, data: Dict[str, Any]) -> bool:
        """Save predictive maintenance data"""
        try:
            predicted_failure_date = data.get("predicted_failure_date")
            if isinstance(predicted_failure_date, str):
                try:
                    predicted_failure_date = datetime.datetime.fromisoformat(predicted_failure_date)
                except:
                    predicted_failure_date = None
            
            analysis_data = data.get("analysis_data")
            if isinstance(analysis_data, dict):
                analysis_data = json.dumps(analysis_data)
            
            with self.session_scope() as session:
                predictive_data = PredictiveMaintenance(
                    timestamp=data.get("timestamp", datetime.datetime.now()),
                    component=data.get("component"),
                    health_score=data.get("health_score"),
                    predicted_failure_date=predicted_failure_date,
                    maintenance_recommended=data.get("maintenance_recommended", False),
                    recommendation=data.get("recommendation"),
                    confidence=data.get("confidence"),
                    analysis_data=analysis_data,
                    maintenance_cost=data.get("maintenance_cost"),
                    failure_cost=data.get("failure_cost")
                )
                session.add(predictive_data)
            return True
        except Exception as e:
            self.add_system_log("error", "Failed to save predictive maintenance data", {"error": str(e)})
            return False
    
    def get_predictive_maintenance(self, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get predictive maintenance data"""
        with self.session_scope() as session:
            if component:
                data = session.query(PredictiveMaintenance).filter_by(component=component).order_by(PredictiveMaintenance.timestamp.desc()).all()
            else:
                data = session.query(PredictiveMaintenance).order_by(PredictiveMaintenance.timestamp.desc()).all()
            return [item.to_dict() for item in data]
    
    def get_component_health(self) -> Dict[str, Any]:
        """Get current health status of all components"""
        with self.session_scope() as session:
            subquery = session.query(
                PredictiveMaintenance.component,
                func.max(PredictiveMaintenance.timestamp).label("max_timestamp")
            ).group_by(PredictiveMaintenance.component).subquery("latest")
            
            latest_scores = session.query(PredictiveMaintenance).join(
                subquery,
                and_(
                    PredictiveMaintenance.component == subquery.c.component,
                    PredictiveMaintenance.timestamp == subquery.c.max_timestamp
                )
            ).all()
            
            return {item.component: item.to_dict() for item in latest_scores}
    
    def get_maintenance_recommendations(self) -> List[Dict[str, Any]]:
        """Get all active maintenance recommendations"""
        with self.session_scope() as session:
            data = session.query(PredictiveMaintenance).filter(
                PredictiveMaintenance.maintenance_recommended == True
            ).order_by(PredictiveMaintenance.timestamp.desc()).all()
            return [item.to_dict() for item in data]
    
    def mark_maintenance_completed(self, component: str) -> bool:
        """Mark maintenance as completed for a component"""
        try:
            with self.session_scope() as session:
                record = session.query(PredictiveMaintenance).filter(
                    PredictiveMaintenance.component == component
                ).order_by(PredictiveMaintenance.timestamp.desc()).first()
                
                if record:
                    new_record = PredictiveMaintenance(
                        component=component,
                        health_score=100,
                        maintenance_recommended=False,
                        recommendation=f"Maintenance completed on {datetime.datetime.now().date()}",
                        confidence=1.0,
                        analysis_data=json.dumps({"maintenance_performed": True})
                    )
                    session.add(new_record)
                    return True
                return False
        except Exception as e:
            self.add_system_log("error", f"Failed to mark maintenance completed for {component}", {"error": str(e)})
            return False

    # System Health Methods
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        with self.session_scope() as session:
            components = self.get_component_health()
            battery_status = self.get_latest_power_data().get("battery_soc", 0)
            alert_count = session.query(SystemLog).filter(SystemLog.log_type == "alert").count()
            
            return {
                "component_health": components,
                "battery_status": battery_status,
                "active_alerts": alert_count,
                "overall_status": "good" if battery_status > 30 and alert_count == 0 else "warning"
            }

    # Data Management Methods
    def truncate_old_data(self, days_to_keep: int = 90) -> Tuple[int, int, int]:
        """Remove old data to prevent database from growing too large"""
        with self.session_scope() as session:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            power_deleted = session.query(PowerData).filter(PowerData.timestamp < cutoff_date).delete()
            weather_deleted = session.query(WeatherData).filter(WeatherData.timestamp < cutoff_date).delete()
            maintenance_deleted = session.query(PredictiveMaintenance).filter(PredictiveMaintenance.timestamp < cutoff_date).delete()
            return (power_deleted, weather_deleted, maintenance_deleted)
    
    def export_data(self, data_type: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        """Export data of specified type within date range"""
        with self.session_scope() as session:
            if data_type == "power":
                query = session.query(PowerData).filter(
                    PowerData.timestamp >= start_date,
                    PowerData.timestamp <= end_date
                ).order_by(PowerData.timestamp)
            elif data_type == "weather":
                query = session.query(WeatherData).filter(
                    WeatherData.timestamp >= start_date,
                    WeatherData.timestamp <= end_date
                ).order_by(WeatherData.timestamp)
            elif data_type == "logs":
                query = session.query(SystemLog).filter(
                    SystemLog.timestamp >= start_date,
                    SystemLog.timestamp <= end_date
                ).order_by(SystemLog.timestamp)
            else:
                return pd.DataFrame()
            
            return pd.read_sql(query.statement, session.bind)

    # Backup/Restore Methods
    def create_backup(self) -> Dict[str, Any]:
        """Create a backup of critical system data"""
        with self.session_scope() as session:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"system_backup_{timestamp}"
            
            return {
                "metadata": {
                    "backup_name": backup_name,
                    "created_at": datetime.datetime.now().isoformat(),
                    "system_version": "1.0"
                },
                "users": [user.to_dict() for user in session.query(User).all()],
                "settings": [setting.to_dict() for setting in session.query(SystemSetting).all()],
                "battery_specs": [spec.to_dict() for spec in session.query(BatterySpecs).all()],
                "power_data": [data.to_dict() for data in session.query(PowerData).order_by(PowerData.timestamp.desc()).limit(1000).all()],
                "weather_data": [data.to_dict() for data in session.query(WeatherData).order_by(WeatherData.timestamp.desc()).limit(1000).all()],
                "logs": [log.to_dict() for log in session.query(SystemLog).order_by(SystemLog.timestamp.desc()).limit(1000).all()]
            }

# Create singleton instance
db = DatabaseManager()