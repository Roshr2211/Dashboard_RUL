import requests
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np

def get_thingspeak_data(channel_id, api_key):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results=1"
    try:
        print("yes")
        logging.info(f"Sending request to ThingSpeak: {url}")
        response = requests.get(url, timeout=10)  # Add a timeout
        logging.info(f"Received response from ThingSpeak. Status code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(data['feeds'][0])
            return data['feeds'][0]
            
        else:
            logging.error(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
    except requests.Timeout:
        print("oh no")
        logging.error("Request to ThingSpeak timed out")
        return None
    except requests.RequestException as e:
        print("no")
        logging.error(f"Request to ThingSpeak failed: {e}")
        return None

def preprocess_thingspeak_data(data):
    df = pd.DataFrame([{
        'cycle': float(data['field1']),
        'ambient_temperature': float(data['field2']),
        'capacity': float(data['field3']),
        'voltage_measured': float(data['field4']),
        'current_measured': float(data['field5']),
        'temperature_measured': float(data['field6']),
        'current_load': float(data['field7']),
        'voltage_load': float(data['field8']),
        'time': datetime.strptime(data['created_at'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
    }])

    features = ['voltage_measured', 'current_measured', 'temperature_measured', 'current_load', 'voltage_load', 'time']
    df_x = df[features].values

        # Reshape to match the expected input shape (None, 6, 371)
    df_x = np.repeat(df_x, 371).reshape(1, 6, 371)

    logging.info(f"Shape of df_x before scaling: {df_x.shape}")

    scaler = StandardScaler()
    df_x = scaler.fit_transform(df_x.reshape(-1, df_x.shape[-1])).reshape(df_x.shape)

    logging.info(f"Shape of df_x after scaling: {df_x.shape}")

    return df_x, df['capacity'].values