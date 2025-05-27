import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Battery Management | Solar-Wind Hybrid Monitor",
    page_icon="🔋",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import pytz
from data_generator import data_generator
from utils import get_status_color
from weather_api_new import weather_api

BATTERY_DATA_SCHEMA = {
    "timestamps": "datetime64[ns]",
    "battery_voltage": "float64",
    "battery_current": "float64",
    "battery_soc": "float64",
    "battery_temperature": "float64",
    "health_pct": "float64",
    "cycle_count": "int64"
}

def create_empty_battery_data_df():
    """Create an empty DataFrame with the correct structure"""
    return pd.DataFrame({
        "timestamps": pd.Series(dtype='datetime64[ns]'),
        "battery_voltage": pd.Series(dtype='float64'),
        "battery_current": pd.Series(dtype='float64'),
        "battery_soc": pd.Series(dtype='float64'),
        "battery_temperature": pd.Series(dtype='float64'),
        "health_pct": pd.Series(dtype='float64'),
        "cycle_count": pd.Series(dtype='int64')
    })

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = datetime.now(pytz.timezone('Africa/Harare'))
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 300
if 'battery_data' not in st.session_state:
    st.session_state.battery_data = create_empty_battery_data_df()

# Title and description
st.title("Battery Management")
st.write("Monitor and manage your battery system performance and health")

def should_refresh_data():
    if 'battery_data' not in st.session_state:
        return True
    
    battery_data = st.session_state.battery_data
    
    if isinstance(battery_data, pd.DataFrame):
        if battery_data.empty:
            return True
    elif isinstance(battery_data, dict):
        if not battery_data:
            return True
    elif battery_data is None:
        return True
    
    elapsed = (datetime.now(pytz.timezone('Africa/Harare')) - st.session_state.last_refresh_time).total_seconds()
    return elapsed >= st.session_state.refresh_interval

def process_battery_data(raw_data):
    """Safely convert raw data to a validated DataFrame"""
    try:
        df = pd.DataFrame(raw_data)
        for col in BATTERY_DATA_SCHEMA:
            if col not in df.columns:
                df[col] = None
        return df.astype(BATTERY_DATA_SCHEMA)
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return create_empty_battery_data_df()

def refresh_data():
    try:
        current_data = data_generator.generate_current_data()
        
        if isinstance(current_data["battery"], dict):
            new_data = pd.DataFrame([current_data["battery"]])
        else:
            new_data = current_data["battery"].copy()
        
        for col in BATTERY_DATA_SCHEMA:
            if col not in new_data.columns:
                new_data[col] = None
        
        if 'timestamp' in new_data.columns:
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        st.session_state.battery_data = new_data.astype(BATTERY_DATA_SCHEMA)
        st.session_state.last_refresh_time = datetime.now(pytz.timezone('Africa/Harare'))
    except Exception as e:
        st.error(f"Error refreshing data: {str(e)}")
        st.session_state.battery_data = create_empty_battery_data_df()

# UI Controls
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=True)
refresh_interval = st.sidebar.slider("Auto-refresh interval (sec)", 5, 60, 15)

if st.sidebar.button("Refresh Now"):
    refresh_data()

if should_refresh_data():
    refresh_data()

# Get current data with safe defaults
try:
    battery_data = st.session_state.battery_data
    last_refresh = st.session_state.last_refresh_time
    
    # Safely extract values with fallbacks
    def get_safe_value(col, default=0):
        try:
            series = battery_data[col].dropna()
            return float(series.iloc[-1]) if not series.empty else default
        except:
            return default
    
    soc_value = get_safe_value("battery_soc", 50)
    voltage = get_safe_value("battery_voltage", 48)
    current = get_safe_value("battery_current", 0)
    temp = get_safe_value("battery_temperature", 25)
    health = get_safe_value("health_pct", 100)
    cycle_count = int(get_safe_value("cycle_count", 0))
    
except Exception as e:
    st.error(f"Error loading battery data: {str(e)}")
    soc_value, voltage, current, temp, health, cycle_count = 50, 48, 0, 25, 100, 0

# Display last refresh time
st.caption(f"Last updated: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

# Create two columns for the main content
col1, col2 = st.columns([2, 1])

# Left column - Battery SOC and key metrics
with col1:
    # Get SOC color safely
    try:
        soc_color = get_status_color(soc_value, {"green": (60, 100), "yellow": (20, 60), "red": (0, 20)})
    except:
        soc_color = "gray"
    
    # Create SOC gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=soc_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Battery State of Charge (SOC)", 'font': {'size': 24}},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': soc_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': "#ffcccc"},
                {'range': [20, 60], 'color': "#ffffcc"},
                {'range': [60, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': soc_value
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA", size=16)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Battery status indicators
    st.subheader("Current Battery Status")
    
    status_cols = st.columns(4)
    
    # Voltage indicator
    try:
        voltage_color = get_status_color(voltage, {"green": (45, 51), "yellow": (42, 45), "red": (0, 42)})
    except:
        voltage_color = "gray"
    
    with status_cols[0]:
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {'#e6f2ff' if st.session_state.theme == 'light' else '#1a2833'};">
                <h3 style="margin:0;">Voltage</h3>
                <h2 style="margin:0; color: {'green' if voltage_color == 'green' else 'orange' if voltage_color == 'yellow' else 'red'};">
                    {voltage:.2f} V
                </h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Current indicator
    current_status = "Charging" if current > 0 else "Discharging"
    current_color = "green" if current > 0 else "red"
    
    with status_cols[1]:
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {'#e6ffe6' if current > 0 else '#ffe6e6'};">
                <h3 style="margin:0;">Current</h3>
                <h2 style="margin:0; color: {current_color};">
                    {abs(current):.2f} A {current_status}
                </h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Temperature indicator
    try:
        temp_color = get_status_color(temp, {"green": (15, 30), "yellow": (30, 40), "red": (40, 100)})
    except:
        temp_color = "gray"
    
    with status_cols[2]:
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {'#f2e6ff' if st.session_state.theme == 'light' else '#261a33'};">
                <h3 style="margin:0;">Temperature</h3>
                <h2 style="margin:0; color: {'green' if temp_color == 'green' else 'orange' if temp_color == 'yellow' else 'red'};">
                    {temp:.1f} °C
                </h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Health indicator
    try:
        health_color = get_status_color(health, {"green": (80, 100), "yellow": (60, 80), "red": (0, 60)})
    except:
        health_color = "gray"
    
    with status_cols[3]:
        st.markdown(
            f"""
            <div style="padding: 10px; border-radius: 5px; background-color: {'#e8f4ea' if st.session_state.theme == 'light' else '#1e352f'};">
                <h3 style="margin:0;">Health</h3>
                <h2 style="margin:0; color: {'green' if health_color == 'green' else 'orange' if health_color == 'yellow' else 'red'};">
                    {health:.1f}%
                </h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Battery current history
    st.subheader("Battery Current History")
    
    try:
        historical_data = data_generator.get_historical_data(timeframe="day")
        
        if not historical_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data["timestamps"],
                y=historical_data["battery_current"],
                name="Current",
                line=dict(color="#4CAF50", width=2),
                fill="tozeroy",
                fillcolor="rgba(76, 175, 80, 0.2)"
            ))
            
            fig.add_shape(
                type="line",
                x0=historical_data["timestamps"].iloc[0],
                y0=0,
                x1=historical_data["timestamps"].iloc[-1],
                y1=0,
                line=dict(color="gray", width=2, dash="dash")
            )
            
            fig.update_layout(
                height=300,
                xaxis_title="Time",
                yaxis_title="Current (A)",
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
                font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical data available yet.")
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")

# Right column - Battery Health and Additional Info
with col2:
    # Battery health gauge
    st.subheader("Battery Health")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Battery Health"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "green" if health >= 80 else "orange" if health >= 60 else "red"},
            'steps': [
                {'range': [0, 60], 'color': "#ffcccc"},
                {'range': [60, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': health
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Battery specs
    st.subheader("Battery Specifications")
    
    if "battery_specs" not in st.session_state:
        st.session_state.battery_specs = {
            "Type": "Lithium Iron Phosphate (LiFePO4)",
            "Capacity": "20 kWh",
            "Nominal Voltage": "48V",
            "Max Charging Rate": "5 kW",
            "Expected Lifespan": "~3000 cycles",
            "Installation Date": "2023-01-15"
        }
    
    display_specs = st.session_state.battery_specs.copy()
    display_specs["Cycle Count"] = f"{cycle_count}"
    
    spec_col1, spec_col2 = st.columns([3, 1])
    
    with spec_col1:
        df_specs = pd.DataFrame(list(display_specs.items()), columns=["Specification", "Value"])
        st.table(df_specs)
    
    with spec_col2:
        if st.button("Edit Specifications", key="edit_battery_specs"):
            st.session_state.editing_battery_specs = True
    
    if st.session_state.get("editing_battery_specs", False):
        st.subheader("Edit Battery Specifications")
        
        with st.form("battery_specs_form"):
            new_specs = {}
            new_specs["Type"] = st.text_input("Battery Type", value=st.session_state.battery_specs["Type"])
            new_specs["Capacity"] = st.text_input("Capacity (kWh)", value=st.session_state.battery_specs["Capacity"])
            new_specs["Nominal Voltage"] = st.text_input("Nominal Voltage", value=st.session_state.battery_specs["Nominal Voltage"])
            new_specs["Max Charging Rate"] = st.text_input("Max Charging Rate", value=st.session_state.battery_specs["Max Charging Rate"])
            new_specs["Expected Lifespan"] = st.text_input("Expected Lifespan", value=st.session_state.battery_specs["Expected Lifespan"])
            new_specs["Installation Date"] = st.text_input("Installation Date", value=st.session_state.battery_specs["Installation Date"])
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Save Changes")
            with col2:
                cancel = st.form_submit_button("Cancel")
            
            if submit:
                st.session_state.battery_specs = new_specs
                st.session_state.editing_battery_specs = False
                st.rerun()
            
            if cancel:
                st.session_state.editing_battery_specs = False
                st.rerun()
    
    # Maintenance tips
    st.subheader("Maintenance Recommendations")
    
    with st.expander("Battery Care Tips", expanded=True):
        st.markdown("""
        - Keep battery temperature between 15-30°C for optimal performance
        - Avoid frequent deep discharges below 20% SOC
        - Periodic checks for loose connections
        - Monitor for unusual temperature increases
        - Schedule full system inspection every 6 months
        """)
    
    st.info("Next scheduled maintenance: 45 days")

# Advanced Battery Analytics Section
st.subheader("Advanced Battery Analytics")

tab1, tab2 = st.tabs(["Performance Metrics", "Charge-Discharge Cycles"])

with tab1:
    try:
        if not historical_data.empty:
            # Performance metrics calculations
            charging_periods = historical_data[historical_data["battery_current"] > 0]
            discharging_periods = historical_data[historical_data["battery_current"] < 0]
            
            if not charging_periods.empty and not discharging_periods.empty:
                time_factor = 1/60
                energy_in = (charging_periods["battery_current"] * charging_periods["battery_voltage"]).sum() * time_factor
                energy_out = abs((discharging_periods["battery_current"] * discharging_periods["battery_voltage"]).sum() * time_factor)
                efficiency = (energy_out / energy_in) * 100 if energy_in > 0 else 0
            else:
                efficiency = 90
            
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                st.metric("Round-Trip Efficiency", f"{efficiency:.1f}%")
            
            with metric_cols[1]:
                min_soc = historical_data["battery_soc"].min()
                max_dod = 100 - min_soc
                st.metric("Max Depth of Discharge", f"{max_dod:.1f}%")
            
            with metric_cols[2]:
                avg_temp = historical_data["battery_temperature"].mean()
                st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            
            # Voltage vs SOC plot
            st.subheader("Voltage vs. State of Charge")
            
            fig = px.scatter(
                historical_data, 
                x="battery_soc", 
                y="battery_voltage",
                color="battery_current",
                color_continuous_scale="RdYlGn",
                labels={
                    "battery_soc": "State of Charge (%)", 
                    "battery_voltage": "Voltage (V)", 
                    "battery_current": "Current (A)"
                }
            )
            
            fig.update_layout(
                height=400,
                margin=dict(l=60, r=20, t=30, b=60),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
                font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA")
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical data available yet.")
    except Exception as e:
        st.error(f"Error displaying performance metrics: {str(e)}")
        
with tab2:
    try:
        if not historical_data.empty:
            historical_data["current_sign"] = np.sign(historical_data["battery_current"])
            historical_data["sign_change"] = historical_data["current_sign"].diff().fillna(0) != 0
            cycle_transitions = historical_data[historical_data["sign_change"]].copy()
            
            if len(cycle_transitions) > 0:
                st.subheader("Charge-Discharge Cycles")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=historical_data["timestamps"],
                    y=historical_data["battery_soc"],
                    name="State of Charge",
                    line=dict(color="#4CAF50", width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=historical_data["timestamps"],
                    y=historical_data["battery_current"],
                    name="Current",
                    line=dict(color="#FF9800", width=2, dash="dot"),
                    yaxis="y2"
                ))
                
                fig.add_trace(go.Scatter(
                    x=cycle_transitions["timestamps"],
                    y=cycle_transitions["battery_soc"],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="circle"),
                    name="Cycle Transitions"
                ))
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Time",
                    yaxis_title="State of Charge (%)",
                    yaxis2=dict(
                        title="Current (A)",
                        title_font=dict(color="#FF9800"),
                        tickfont=dict(color="#FF9800"),
                        overlaying="y",
                        side="right"
                    ),
                    margin=dict(l=60, r=60, t=30, b=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)' if st.session_state.theme == "dark" else 'rgba(240,242,246,0.5)',
                    font=dict(color="#262730" if st.session_state.theme == "light" else "#FAFAFA"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Cycle statistics
                st.subheader("Cycle Statistics")
                
                charge_cycles = len(cycle_transitions[cycle_transitions["current_sign"] > 0])
                discharge_cycles = len(cycle_transitions[cycle_transitions["current_sign"] < 0])
                
                cycle_cols = st.columns(3)
                
                with cycle_cols[0]:
                    st.metric("Complete Cycles Today", f"{min(charge_cycles, discharge_cycles)}")
                
                with cycle_cols[1]:
                    st.metric("Charging Events", f"{charge_cycles}")
                
                with cycle_cols[2]:
                    st.metric("Discharging Events", f"{discharge_cycles}")
            else:
                st.info("No complete charge-discharge cycles detected.")
        else:
            st.info("Not enough historical data available yet.")
    except Exception as e:
        st.error(f"Error displaying cycle analysis: {str(e)}")