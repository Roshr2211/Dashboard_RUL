import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import logging
import json
import sys
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


# def preprocess_data_to_cycles():
#     path = "Data/"
#     files = [f for f in os.listdir(path) if f.endswith('.mat')]
#     dis_mat = [os.path.join(path, f) for f in files]
#     battery_grp = {}

#     for f in files:
#         key = f.split('.')[0]
#         battery_grp[key] = key

#     bs = [f.split('.')[0] for f in files]

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
#         Cycles[bs[i]] = {'count': 0}
#         for param in params:
#             Cycles[bs[i]][param] = [datas[i][j][param][0][0][0] for j in range(datas[i].size) if types[i][j] == 'discharge']

#         cap, amb_temp, dis, cycle_counts = [], [], [], []
#         total_cycles = 0
#         discharge_number = 0

#         for j in range(datas[i].size):
#             total_cycles += 1
#             if types[i][j] == 'discharge':
#                 discharge_number += 1
#                 dis.append(discharge_number)
#                 cap.append(datas[i][j]['Capacity'][0][0][0])
#                 amb_temp.append(ambient_temperatures[i][j])
#                 cycle_counts.append(total_cycles)

#         Cycles[bs[i]]['Capacity'] = np.array(cap)
#         Cycles[bs[i]]['ambient_temperatures'] = np.array(amb_temp)
#         Cycles[bs[i]]['discharge_number'] = np.array(dis)
#         Cycles[bs[i]]['Cycle_count'] = np.array(cycle_counts)

#     Cycles = pd.DataFrame(Cycles)
#     return Cycles


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
def generate_data(exp):
    Cycles = preprocess_data_to_cycles()
    df_all = pd.DataFrame({})
    exp_try_out = {exp}

    for bat in exp_try_out:
        if bat not in Cycles.columns:
            print(f"Battery {bat} not found in Cycles DataFrame")
            continue

        df = pd.DataFrame({})
        cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity', 'ambient_temperatures', 'SOC', 'SOH', 'Cycle_count', 'discharge_number']
        for col in cols:
            df[col] = Cycles[bat][col]
        df_all = pd.concat([df_all, df], ignore_index=True)

    df = df_all.reset_index(drop=True)
    return df
# def generate_data(exp):
#     path = "Data/"
#     file = f"{exp}.mat"
#     file_path = os.path.join(path, file)
    
#     if not os.path.exists(file_path):
#         print(f"Battery {exp} not found")
#         return pd.DataFrame()

#     data = loadmat(file_path)
#     x = data[exp]["cycle"][0][0][0]
    
#     types = x['type']
#     ambient_temperatures = list(map(lambda y: y[0][0], x['ambient_temperature']))
#     datas = x['data']

#     df = pd.DataFrame()
#     cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time', 'Capacity']
    
#     for col in cols:
#         df[col] = [datas[j][col][0][0][0] for j in range(datas.size) if types[j] == 'discharge']

#     df['ambient_temperatures'] = [ambient_temperatures[j] for j in range(datas.size) if types[j] == 'discharge']
#     df['Cycle_count'] = range(1, len(df) + 1)
#     df['discharge_number'] = range(1, len(df) + 1)

#     initial_capacity = df['Capacity'].iloc[0]
#     df['SOC'] = df.apply(lambda row: calculate_soc(row['Current_measured'], row['Time'], initial_capacity), axis=1)
#     df['SOH'] = df['Capacity'] / initial_capacity

#     return df

def generate_plot(df):
    plt.plot(df['Cycle_count'], df['Capacity'])
    plt.xlabel('Cycle Count')
    plt.ylabel('Capacity')
    plt.title('Capacity vs. Cycle Count')
    plt.savefig('public/plot.png')

logging.basicConfig(level=logging.DEBUG, filename='python_script.log')

# ... (rest of the imports and functions)

if __name__ == "__main__":
    try:
        default_experiment = "B0005"
        
        if len(sys.argv) > 1:
            experiment = sys.argv[1]
        else:
            logging.warning(f"No experiment argument provided. Using default: {default_experiment}")
            experiment = default_experiment

        logging.info(f"Processing experiment: {experiment}")
        df = generate_data(experiment)
        
        if df.empty:
            logging.error(f"No data found for experiment {experiment}")
            result = {"error": f"No data found for experiment {experiment}"}
        else:
            generate_plot(df)
            result = json.loads(df.to_json(orient='records'))
        
        # Print the result as a single JSON string
        print(json.dumps({"result": result}))
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)