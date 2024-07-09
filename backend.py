import os
import logging
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from flask import Flask, request, jsonify

# Define TransformerBlock
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Register the MSE loss function
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)

# Load the pre-trained model
# model = keras.models.load_model('rul_prediction_model_transformer.h5', custom_objects={'TransformerBlock': TransformerBlock, 'mse': mse})

try:
    model = keras.models.load_model('rul_prediction_model_transformer.h5', custom_objects={'TransformerBlock': TransformerBlock,'mse': mse})
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

def to_padded_numpy(l, shape):
    padded_array = np.zeros(shape)
    padded_array[:len(l)] = l
    return padded_array

def calculate_soc(current, time, initial_capacity):
    # Simple coulomb counting method
    charge = np.trapz(current, time)
    soc = 1 - (charge / initial_capacity)
    return np.clip(soc, 0, 1)  # Ensure SOC is between 0 and 1

def calculate_soh(current_capacity, initial_capacity):
    return current_capacity / initial_capacity

def preprocess_data_to_cycles():
    path = "Data/"
    files = [f for f in os.listdir(path) if f.endswith('.mat')]
    dis_mat = [os.path.join(path, f) for f in files]
    battery_grp = {}

    for f in files:
        key = f.split('.')[0]
        battery_grp[key] = key

    bs = [f.split('.')[0] for f in files]

    ds = []
    for f in dis_mat:
        ds.append(loadmat(f))

    types = []
    times = []
    ambient_temperatures = []
    datas = []

    for i in range(len(ds)):
        x = ds[i][bs[i]]["cycle"][0][0][0]
        ambient_temperatures.append(list(map(lambda y: y[0][0], x['ambient_temperature'])))
        types.append(x['type'])
        times.append(x['time'])
        datas.append(x['data'])

    batteries = []
    cycles = []
    for i in range(len(ds)):
        batteries.append(bs[i])
        cycles.append(datas[i].size)

    battery_cycle_df = pd.DataFrame({'Battery': batteries, 'Cycle': cycles}).sort_values('Battery', ascending=True)
    battery_cycle_df.drop_duplicates(inplace=True)

    Cycles = {}
    params = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']

    for i in range(len(bs)):
        Cycles[bs[i]] = {'count': 0}
        for param in params:
            Cycles[bs[i]][param] = [datas[i][j][param][0][0][0] for j in range(datas[i].size) if types[i][j] == 'discharge']

        cap, amb_temp, dis, cycle_counts = [], [], [], []
        total_cycles = 0
        discharge_number = 0

        for j in range(datas[i].size):
            total_cycles += 1
            if types[i][j] == 'discharge':
                discharge_number += 1
                dis.append(discharge_number)
                cap.append(datas[i][j]['Capacity'][0][0][0])
                amb_temp.append(ambient_temperatures[i][j])
                cycle_counts.append(total_cycles)

        Cycles[bs[i]]['Capacity'] = np.array(cap)
        Cycles[bs[i]]['ambient_temperatures'] = np.array(amb_temp)
        Cycles[bs[i]]['discharge_number'] = np.array(dis)
        Cycles[bs[i]]['Cycle_count'] = np.array(cycle_counts)

        initial_capacity = Cycles[bs[i]]['Capacity'][0]
        Cycles[bs[i]].update({
            'SOC': [],
            'SOH': [],
        })

        for j in range(len(Cycles[bs[i]]['Capacity'])):
            current = np.array(Cycles[bs[i]]['Current_measured'][j])
            time = np.array(Cycles[bs[i]]['Time'][j])
            capacity = Cycles[bs[i]]['Capacity'][j]

            soc = calculate_soc(current, time, initial_capacity)
            soh = calculate_soh(capacity, initial_capacity)

            Cycles[bs[i]]['SOC'].append(soc)
            Cycles[bs[i]]['SOH'].append(soh)

    Cycles = pd.DataFrame(Cycles)

    return Cycles


# def preprocess_data_to_cycles():
#     path = "Data/"
#     logging.info(f"Looking for .mat files in: {path}")
#     files = [f for f in os.listdir(path) if f.endswith('.mat')]
#     logging.info(f"Found {len(files)} .mat files: {files}")
    
#     dis_mat = [os.path.join(path, f) for f in files]
#     battery_grp = {}

#     for f in files:
#         key = f.split('.')[0]
#         battery_grp[key] = key

#     bs = [f.split('.')[0] for f in files]
#     logging.info(f"Processed battery names: {bs}")

#     ds = []
#     for f in dis_mat:
#         ds.append(loadmat(f))

#     types = []
#     times = []
#     ambient_temperatures = []
#     datas = []

#     for i in range(len(ds)):
#         x = ds[i][bs[i]]["cycle"][0][0][0]
#         ambient_temperatures.append(list(map(lambda y: y[0][0], x['ambient_temperature'])))
#         types.append(x['type'])
#         times.append(x['time'])
#         datas.append(x['data'])

#     batteries = []
#     cycles = []
#     for i in range(len(ds)):
#         batteries.append(bs[i])
#         cycles.append(datas[i].size)

#     battery_cycle_df = pd.DataFrame({'Battery': batteries, 'Cycle': cycles}).sort_values('Battery', ascending=True)
#     battery_cycle_df.drop_duplicates(inplace=True)

#     Cycles = {}
#     params = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']

#     for i in range(len(bs)):
#         Cycles[bs[i]] = {}
#         Cycles[bs[i]]['count'] = 0
#         for param in params:
#             Cycles[bs[i]][param] = []
#             for j in range(datas[i].size):
#                 if types[i][j] == 'discharge':
#                     Cycles[bs[i]][param].append(datas[i][j][param][0][0][0])

#         cap = []
#         amb_temp = []
#         for j in range(datas[i].size):
#             if types[i][j] == 'discharge':
#                 cap.append(datas[i][j]['Capacity'][0][0][0])
#                 amb_temp.append(ambient_temperatures[i][j])

#         Cycles[bs[i]]['Capacity'] = np.array(cap)
#         Cycles[bs[i]]['ambient_temperatures'] = np.array(amb_temp)

#     Cycles = pd.DataFrame(Cycles)
#     logging.info(f"Cycles DataFrame columns: {Cycles.columns}")
#     return Cycles

def get_exp_based_df(exp):
    Cycles = preprocess_data_to_cycles()
    df_all = pd.DataFrame({})
    max_len = 0

    exp_try_out = exp
    logging.info(f"Processing experiment: {exp_try_out}")

    for bat in exp_try_out:
        if bat not in Cycles.columns:
            logging.warning(f"Battery {bat} not found in Cycles DataFrame. Available batteries: {Cycles.columns}")
            continue

        df = pd.DataFrame({})
        cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity', 'ambient_temperatures']
        for col in cols:
            if col not in Cycles[bat]:
                logging.warning(f"Column {col} not found for battery {bat}")
                continue
            df[col] = Cycles[bat][col]
        
        if df.empty:
            logging.warning(f"No data found for battery {bat}")
            continue

        max_l = np.max(df['Time'].apply(lambda x: len(x)).values)
        max_len = max(max_l, max_len)
        df_all = pd.concat([df_all, df], ignore_index=True)

    if df_all.empty:
        raise ValueError("No data could be processed for the given experiment")

    df = df_all.reset_index(drop=True)

    for i, j in enumerate(df['Capacity']):
        try:
            if len(j):
                df['Capacity'][i] = j[0]
            else:
                df['Capacity'][i] = 0
        except:
            pass

    df_x = df.drop(columns=['Capacity', 'ambient_temperatures']).values
    df_y = df['Capacity'].values

    n, m = df_x.shape[0], df_x.shape[1]
    temp2 = np.zeros((n, m, max_len))
    for i in range(n):
        for j in range(m):
            temp2[i][j] = to_padded_numpy(df_x[i][j], max_len)

    df_x = temp2
    return df_x, df_y

@app.route('/')
def home():
    return "RUL Prediction API is running"

# @app.route('/predict', methods=['POST'])
# def predict():
    # data = request.get_json()
    # experiment = data['experiment']
    # df_x, df_y = get_exp_based_df(experiment)
    
    # # Normalize the data
    # scaler = StandardScaler()
    # df_x = scaler.fit_transform(df_x.reshape(-1, df_x.shape[-1])).reshape(df_x.shape)
    
    # predictions = model.predict(df_x)
    
    # return jsonify(predictions=predictions.flatten().tolist(), true_values=df_y.tolist())

# @app.route('/data', methods=['POST'])
# def generate_data():
#     logging.info("Starting data generation process")
#     try:
#         Cycles = preprocess_data_to_cycles()
#         logging.info(f"Preprocessed data. Cycles DataFrame shape: {Cycles.shape}")

#         df_all = pd.DataFrame({})
#         exp_try_out = ['B0005']
#         logging.info(f"Generating data for experiments: {exp_try_out}")

#         for bat in exp_try_out:
#             if bat not in Cycles.columns:
#                 logging.warning(f"Battery {bat} not found in Cycles DataFrame")
#                 continue

#             df = pd.DataFrame({})
#             cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity', 'ambient_temperatures', 'SOC', 'SOH', 'Cycle_count', 'discharge_number']
#             for col in cols:
#                 if col not in Cycles[bat]:
#                     logging.warning(f"Column {col} not found for battery {bat}")
#                     continue
#                 df[col] = Cycles[bat][col]
            
#             logging.info(f"Generated data for battery {bat}. Shape: {df.shape}")
#             df_all = pd.concat([df_all, df], ignore_index=True)

#         df = df_all.reset_index(drop=True)
#         logging.info(f"Final DataFrame shape: {df.shape}")

#         # Convert DataFrame to JSON
#         json_data = df.to_json(orient='records')
        
#         logging.info("Data generation completed successfully")
#         return jsonify({"data": json_data})

#     except Exception as e:
#         logging.error(f"Error in generate_data: {str(e)}", exc_info=True)
#         return jsonify({"error": "An error occurred during data generation"}), 500
# def generate_plot(df):
#     plt.plot(df['Cycle_count'], df['Capacity'])
#     plt.xlabel('Cycle Count')
#     plt.ylabel('Capacity')
#     plt.title('Capacity vs. Cycle Count')
#     plt.savefig('plot.png')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        experiment = data.get('experiment')
        if not experiment:
            raise ValueError("Missing 'experiment' in request data")

        logging.info(f"Processing experiment: {experiment}")
        df_x, df_y = get_exp_based_df(experiment)

        logging.info(f"Data shape after processing - df_x: {df_x.shape}, df_y: {df_y.shape}")

        # Normalize the data
        scaler = StandardScaler()
        df_x = scaler.fit_transform(df_x.reshape(-1, df_x.shape[-1])).reshape(df_x.shape)

        logging.info("Making predictions...")
        predictions = model.predict(df_x)
        
        response = jsonify(predictions=predictions.flatten().tolist(), true_values=df_y.tolist())
        logging.debug(f"Response: {response}")

        return response
    except ValueError as ve:
        logging.error(f"ValueError in prediction: {ve}")
        return jsonify(error=str(ve)), 400
    except Exception as e:
        logging.error(f"Unexpected error in prediction: {e}", exc_info=True)
        return jsonify(error="An unexpected error occurred. Please check the logs for more information."), 500

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("debug.log"),
            logging.StreamHandler()
        ]
    )
    app.run(host='0.0.0.0', port=5000)
