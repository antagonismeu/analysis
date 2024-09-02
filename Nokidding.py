import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

multiplier = 1
seq_length = 7

model = load_model(f'models/RNN{multiplier}x.h5')

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['lq_total_qtrly_wages']).values
    y = data['lq_total_qtrly_wages'].values

    X = X * multiplier
    y = y * multiplier

    return X, y

def create_sequences(X, y, seq_length):
    num_samples = len(X)
    
    sequences = []
    labels = []

    for i in range(num_samples):
        seq = np.tile(X[i], (seq_length, 1))
        sequences.append(seq)
        
        label = np.mean(y[i])
        labels.append(label)
    
    return np.array(sequences), np.array(labels)

X, y = load_data_from_csv('no_kidding.csv')
X_seq, y_seq = create_sequences(X, y, seq_length)

y_pred = model.predict(X_seq)
y_pred_avg = np.mean(y_pred, axis=1)
print(y_pred.shape)

mae = mean_absolute_error(y_seq, y_pred_avg)
print(f"Mean Absolute Error: {mae}")

# Replotting with True Values, Predicted Values, and Absolute Errors
plt.figure(figsize=(12, 8))

plt.plot(range(1, len(y_seq) + 1), y_seq, marker='o', linestyle='-', color='green', label='True Values')
plt.plot(range(1, len(y_pred_avg) + 1), y_pred_avg, marker='x', linestyle='-', color='red', label='Predicted Values')

plt.title('True Values Verserse Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('images/accuracy_RNN.png')

# Display the plot
plt.show()

# Print sample values
for i, (pred, actual) in enumerate(zip(y_pred_avg, y_seq)):
    print(f"Sample {i + 1}: Predicted = {pred}, Actual = {actual}")