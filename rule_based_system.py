import pandas as pd

# Load the dataset
dataset = pd.read_csv("rule_based_anomaly_dataset.csv")

# Define thresholds for anomalies
voltage_threshold = 3.0  # Voltage threshold
temperature_threshold = 90.0  # Temperature threshold
humidity_threshold = 100.0  # Humidity threshold

# Apply rules to detect anomalies
dataset['Predicted_Anomaly'] = (
    (dataset['Voltage'] > voltage_threshold) |
    (dataset['Temperature'] > temperature_threshold) |
    (dataset['Humidity'] >= humidity_threshold)
).astype(int)

# Evaluate the rule-based system
true_anomalies = dataset['Anomaly']
predicted_anomalies = dataset['Predicted_Anomaly']

# Accuracy calculation
accuracy = (true_anomalies == predicted_anomalies).mean() * 100
print(f"Rule-Based System Accuracy: {accuracy:.2f}%")

# Save the dataset with predictions
dataset.to_csv("rule_based_anomaly_results.csv", index=False)

print("Results saved as 'rule_based_anomaly_results.csv'.")