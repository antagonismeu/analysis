import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

multiplier = 1
model = load_model(f'models/DL{multiplier}x.h5')

def detransform_log(y):
    return y / multiplier

def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['lq_total_qtrly_wages']).values
    y = data['lq_total_qtrly_wages'].values
    print(X.shape, y.shape)
    return X * multiplier, y * multiplier

def get_input_data(file_path):
    data = pd.read_csv(file_path)

    input_data = data.values[0]  
    
    return input_data.reshape(1, -1) * multiplier

def predict_and_plot(input_data, true_values):
    predictions = model.predict(input_data)
    predictions_detransformed = detransform_log(predictions)
    
    # Plotting True Values vs Predictions
    plt.figure(figsize=(12, 8))
    
    plt.plot(range(1, len(true_values) + 1), true_values / multiplier, marker='o', linestyle='-', color='green', label='True Values')
    plt.plot(range(1, len(predictions_detransformed) + 1), predictions_detransformed, marker='x', linestyle='-', color='red', label='Predicted Values')
    
    plt.title('True Values vs Predicted Values')
    plt.xlabel('Sample Index')
    plt.ylabel('lq_total_qtrly_wages')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'images/accuracy_DL{multiplier}x.png')
    
    # Display the plot
    plt.show()
    
    return predictions_detransformed

while True:
    user_input = input("Continue with new input? (Press 'q' to quit, 'csv' to load from CSV, any other key to enter manually): ")
    
    if user_input.lower() == 'q':
        print("Exiting the program.")
        break
    
    elif user_input.lower() == 'csv':
        file_path = input("Enter the path to the CSV file: ")
        X, y = load_data_from_csv(file_path)
        predictions = predict_and_plot(X, y)
        
       
        for i, (pred, actual) in enumerate(zip(predictions, y)):
            print(f"Sample {i + 1}: Predicted = {pred[0]:.4f}, Actual = {actual / multiplier:.4f}")
        
    else:
        file_path = input("Enter the path to the input CSV file: ")
        input_data = get_input_data(file_path)
        prediction = model.predict(input_data)
        detransformed_prediction = detransform_log(prediction)[0][0]
        print(f"Predicted lq_total_qtrly_wages: {detransformed_prediction:.4f}")