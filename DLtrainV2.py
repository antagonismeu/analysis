import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Normalization, Dropout
from tensorflow.keras.models import Model
import numpy as np


data = pd.read_csv("filtered_data.csv")


X = data.drop(columns=['lq_total_qtrly_wages']).values
y = data['lq_total_qtrly_wages'].values

multiplier = 1

y = y * multiplier  
X = X * multiplier

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


input_features = Input(shape=(X_train.shape[1],), name='input_features')


x = Dense(128, activation='relu')(input_features)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear', name='lq_total_qtrly_wages')(x)


model = Model(inputs=input_features, outputs=output)


model.compile(optimizer='adam', loss='mse', metrics=['mae'])


model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=500, restore_best_weights=True)])


model.evaluate(X_val, y_val)


model.save(f'models/DL{multiplier}x.h5')

"""
accuracy rank, Error range
0.0~2.0, +-0.03
3.0~9.0 +-3.5
>=10.0 +-9.0
"""