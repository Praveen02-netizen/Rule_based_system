import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the generated dataset
data = pd.read_csv('sensor_data.csv')

# Extract true anomalies
true_anomalies = data['Anomaly']

# Define thresholds for rule-based system
voltage_threshold = 3.0
temperature_threshold = 45
humidity_threshold = 85

# Predict anomalies based on rule-based system
predicted_anomalies = (
    (data['Voltage'] > voltage_threshold) |
    (data['Temperature'] > temperature_threshold) |
    (data['Humidity'] > humidity_threshold)
).astype(int)

# Compare true anomalies with predicted anomalies
print("Confusion Matrix:")
print(confusion_matrix(true_anomalies, predicted_anomalies))

print("\nClassification Report:")
print(classification_report(true_anomalies, predicted_anomalies))

print("\nAccuracy Score:")
print(accuracy_score(true_anomalies, predicted_anomalies))
