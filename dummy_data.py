import numpy as np
import pandas as pd
import requests
import time

# ThingSpeak credentials
api_key = ''  # Replace with your actual ThingSpeak API key
channel_url = f'https://api.thingspeak.com/update?api_key={api_key}'

# Function to generate dummy data
def generate_dummy_data():
    np.random.seed()  # Seed with system time
    cycles = np.ones(1)
    ambient_temperature = np.random.normal(loc=25, scale=5, size=1)
    datetime = pd.date_range(start=pd.Timestamp.now(), periods=1, freq='H').strftime('%d-%m-%Y %H:%M')
    capacity = np.random.uniform(1.5, 2.5, size=1)
    voltage_measured = np.random.uniform(3.0, 4.2, size=1)
    current_measured = np.random.uniform(-2.0, 2.0, size=1)
    temperature_measured = np.random.uniform(20, 30, size=1)
    current_load = np.random.uniform(-2.0, 2.0, size=1)
    voltage_load = np.random.uniform(0, 5, size=1)
    time_value = np.random.uniform(0, 10, size=1)

    data = pd.DataFrame({
        'cycle': cycles,
        'ambient_temperature': ambient_temperature,
        'datetime': datetime,
        'capacity': capacity,
        'voltage_measured': voltage_measured,
        'current_measured': current_measured,
        'temperature_measured': temperature_measured,
        'current_load': current_load,
        'voltage_load': voltage_load,
        'time': time_value
    })
    
    return data

# Function to send data to ThingSpeak
def send_data_to_thingspeak(data):
    for _, row in data.iterrows():
        response = requests.post(channel_url, data={
            'field1': row['cycle'],
            'field2': row['ambient_temperature'],
            'field3': row['capacity'],
            'field4': row['voltage_measured'],
            'field5': row['current_measured'],
            'field6': row['temperature_measured'],
            'field7': row['current_load'],
            'field8': row['voltage_load']
        })
        if response.status_code == 200:
            print('Data successfully sent to ThingSpeak')
        else:
            print('Failed to send data to ThingSpeak')
        time.sleep(15)  # ThingSpeak's rate limit (update every 15 seconds)

# Continuous data generation and sending
while True:
    dummy_data = generate_dummy_data()
    send_data_to_thingspeak(dummy_data)
    time.sleep(15)  # Wait for a bit before generating new data to match ThingSpeak's rate limit
