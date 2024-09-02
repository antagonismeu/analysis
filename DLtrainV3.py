import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Bidirectional, GRU, Dense, Dropout, Input
from tensorflow.keras.models import Model

multiplier = 1
data = pd.read_csv('filtered_data.csv')

X = data.drop(columns=['lq_total_qtrly_wages']).values
y = data['lq_total_qtrly_wages'].values

X = X * multiplier
y = y * multiplier



def create_sequences(X, y, seq_length):
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)  

    sequences = []
    labels = []
    
    for start in range(0, num_samples - seq_length + 1, seq_length):
        selected_indices = indices[start:start + seq_length]
        if len(selected_indices) == seq_length:
            sequences.append(X[selected_indices])
            labels.append(y[selected_indices[-1]])  
    
    return np.array(sequences), np.array(labels)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


seq_length = 7
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)

inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
x = Bidirectional(GRU(50, return_sequences=True))(inputs)
x = Dropout(0.2)(x)
x = Bidirectional(GRU(50))(x)
x = Dropout(0.2)(x)
outputs = Dense(seq_length)(x)

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train_seq, y_train_seq, epochs=500, batch_size=32, validation_data=(X_val_seq, y_val_seq), 
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True)])

model.evaluate(X_val_seq, y_val_seq)

model.save(f'models/RNN{multiplier}x.h5')

y_pred = model.predict(X_val_seq)
print(y_pred)
print(y_val_seq)