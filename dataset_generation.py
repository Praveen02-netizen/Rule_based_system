import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Dataset size
num_samples = 1000
anomaly_ratio = 0.1  # 10% anomalies

# Normal data generation
voltage_normal = np.random.uniform(0.5, 3.0, int(num_samples * (1 - anomaly_ratio)))  # Normal voltage
temperature_normal = np.random.uniform(20, 90, int(num_samples * (1 - anomaly_ratio)))  # Normal temperature
humidity_normal = np.random.uniform(20, 99, int(num_samples * (1 - anomaly_ratio)))  # Normal humidity

# Anomalous data generation
voltage_anomalous = np.random.uniform(3.1, 5.0, int(num_samples * anomaly_ratio))  # Voltage anomalies
temperature_anomalous = np.random.uniform(91, 120, int(num_samples * anomaly_ratio))  # Temperature anomalies
humidity_anomalous = np.random.uniform(100, 101, int(num_samples * anomaly_ratio))  # Humidity anomalies

# Combine normal and anomalous data
voltage = np.concatenate([voltage_normal, voltage_anomalous])
temperature = np.concatenate([temperature_normal, temperature_anomalous])
humidity = np.concatenate([humidity_normal, humidity_anomalous])

# Create anomaly labels (0 for normal, 1 for anomaly)
anomaly_voltage = np.concatenate([np.zeros(len(voltage_normal)), np.ones(len(voltage_anomalous))]).astype(bool)
anomaly_temperature = np.concatenate([np.zeros(len(temperature_normal)), np.ones(len(temperature_anomalous))]).astype(bool)
anomaly_humidity = np.concatenate([np.zeros(len(humidity_normal)), np.ones(len(humidity_anomalous))]).astype(bool)

# Overall anomaly label (1 if any parameter is anomalous)
anomaly = (anomaly_voltage | anomaly_temperature | anomaly_humidity).astype(int)

# Shuffle the dataset
indices = np.arange(num_samples)
np.random.shuffle(indices)
voltage = voltage[indices]
temperature = temperature[indices]
humidity = humidity[indices]
anomaly = anomaly[indices]

# Create a DataFrame
dataset = pd.DataFrame({
    "Voltage": voltage,
    "Temperature": temperature,
    "Humidity": humidity,
    "Anomaly": anomaly
})

# Save the dataset to a CSV file
dataset.to_csv("rule_based_anomaly_dataset.csv", index=False)

print("Dataset created and saved as 'rule_based_anomaly_dataset.csv'.")
