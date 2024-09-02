import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

multiplier = 1
seq_length = 7

model = load_model(f'models/RNN{multiplier}x.h5')

def get_input_data():
    print("Please enter the 7x9 matrix of inputs, row by row:")
    
    input_data = []
    for i in range(seq_length):
        row = input(f"Row {i + 1} (enter 9 values separated by commas): ")
        row_data = list(map(float, row.split(',')))
        if len(row_data) != 9:
            raise ValueError("Each row must contain exactly 9 values.")
        input_data.append(row_data)
    
    return np.array(input_data).reshape(1, seq_length, 9)

while True:
    input_data = get_input_data()
    
    if input_data is None:
        print("Terminating the procedure.")
        break

    
    input_data *= multiplier

    
    predicted_wages = model.predict(input_data)

    
    print(f"Predicted lq_total_qtrly_wages: {predicted_wages.flatten()}")

    
    cont = input("Press 'q' to quit or any other key to continue: ")
    if cont.lower() == 'q':
        print("Terminating the procedure.")
        break
