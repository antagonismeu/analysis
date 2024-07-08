import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GRU, Layer
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import argparse


'''
Coattention + self-attention + GRU
Version = 1.0.0
'''
BATCH_SIZE = 6

def configuration():
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

class CoAttention(Layer):
    def __init__(self, units):
        super(CoAttention, self).__init__()
        self.Wq = Dense(units)
        self.Wk = Dense(units)
        self.Wv = Dense(units)

    def call(self, features1, features2):
        query = self.Wq(features1)
        key = self.Wk(features2)
        value = self.Wv(features2)
        score = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, value)
        return context_vector

class FeatureAttention(Layer):
    def __init__(self, units):
        super(FeatureAttention, self).__init__()
        self.dense = Dense(units, activation='tanh')
        self.context_vector = Dense(1, activation='softmax')

    def call(self, features):
        attention_weights = self.context_vector(self.dense(features))
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

class CustomModel(tf.keras.Model):
    def __init__(self, number_of_departments, coattention_units, feature_attention_units, gru_units, input_dim, output_dim):
        super(CustomModel, self).__init__()
        self.coattention_layer = CoAttention(coattention_units)
        self.embedding1 = Embedding(input_dim, output_dim)
        self.embedding2 = Embedding(input_dim, output_dim)
        self.feature_attention_layer_processing = FeatureAttention(feature_attention_units)
        self.feature_attention_layer_total = FeatureAttention(feature_attention_units)
        self.gru_layer = GRU(gru_units, return_sequences=True)
        self.dense_layer = Dense(number_of_departments, activation='softmax')
        self.weight = self.add_weight(shape=(BATCH_SIZE, 16), 
                                      initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1), 
                                      trainable=True)
        self.precision = Precision()
        self.recall = Recall()
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, att_epochs=3):
        processing_data, cost_data = inputs
        addition = self.gru_layer(self.embedding1(processing_data))
        interpolate = processing_data
        for _ in range(att_epochs):
            coattention_output = self.coattention_layer(self.embedding1(interpolate), self.embedding2(cost_data))
            interpolate = self.feature_attention_layer_processing(coattention_output)
        feature_attention_output_processing  = interpolate
        for _ in range(att_epochs):
            x = self.weight * feature_attention_output_processing + (tf.ones_like(self.weight) - self.weight) * addition
            feature_attention_output_processing = self.feature_attention_layer_total(x)
        output = self.dense_layer(feature_attention_output_processing)
        return output

    def train_step(self, processing_data, cost_data, labels):
        with tf.GradientTape() as tape:
            predictions = self([processing_data, cost_data])
            loss = self.loss_fn(labels, predictions)
            self.precision.update_state(labels, predictions)
            self.recall.update_state(labels, predictions)
            print(f"Loss: {loss.numpy()}, Precision: {self.precision.result().numpy()}, Recall: {self.recall.result().numpy()}")
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

def main(relay, total):
    configuration()
    feature_dim = 128
    number_of_departments = 10
    attention_units = 128
    feature_attention_units = 64
    gru_units = 64
    input_dim = 128
    checkpoints = relay
    output_dim = 128
    log_path = './logs/CM.log'
    batch_size = 16
    epochs = total
    model = CustomModel(number_of_departments, attention_units, feature_attention_units, gru_units, input_dim, output_dim)
    tensorboard_callback = TensorBoard(log_dir="./logs")
    checkpoint_callback = ModelCheckpoint(filepath='./models/best_model.h5', save_best_only=True, monitor='val_loss')

    for epoch in range(epochs):
        processing_data = tf.random.normal((batch_size, 10))
        cost_data = tf.random.normal((batch_size, 10))
        labels = tf.random.uniform((batch_size, 10), maxval=2, dtype=tf.int32)
        labels = tf.cast(labels, tf.float32)

        loss = model.train_step(processing_data, cost_data, labels)
        with open(log_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}, Loss: {loss.numpy()}\n")
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")
        if (epoch + 1) % checkpoints == 0 or (epoch + 1) == epochs:
            model.save_weights(f'models/Model{epoch+1}')
    model.summary()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify the total loops and the saving frequency')
    parser.add_argument('-total', type=int, default=1, help='Total number of epochs')
    parser.add_argument('-freq', type=int, default=1, help='Checkpoint saving frequency')
    args = parser.parse_args()
    main(args.freq, args.total)