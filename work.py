import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding, Flatten
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import precision_score
import numpy as np

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, batch_size, seq_len, q, k, v, mask_flag):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask_flag == 'decoder':
            mask = tf.cast(tf.random.uniform((batch_size, 1, 1, seq_len)) > 0.5, tf.float32)
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        return output, attention_weights

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(batch_size, seq_len, q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights
    

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model

    def get_angles(self, pos, i, d_model):
        angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return angles

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                     np.arange(self.d_model)[np.newaxis, :],
                                     self.d_model)
        
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return inputs + tf.cast(pos_encoding, tf.float32)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model=key_dim, num_heads=num_heads)
        self.ffn = [
            Dense(ff_dim, activation='relu'),
            Dense(key_dim)
        ]
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        for layer  in self.ffn :        
            out1 = layer(out1)
        ffn_output = self.dropout2(out1, training=training)
        return self.layernorm2(out1 + ffn_output)



class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model=key_dim, num_heads=num_heads)
        self.mha2 = MultiHeadAttention(d_model=key_dim, num_heads=num_heads)
        self.ffn = [
            Dense(ff_dim, activation='relu'),
            Dense(key_dim)
        ]
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
    
    def call(self, x, enc_output, training, mask='decoder'):
        attn1, _ = self.mha1(x, x, x, mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        
        attn2, _ = self.mha2(out1, enc_output, enc_output)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        for layer in self.ffn :
            out2 = layer(out2)
        ffn_output = self.dropout3(out2, training=training)
        return self.layernorm3(out2 + ffn_output)


class TransformerModel(tf.keras.Model):
    def __init__(self, num_heads, key_dim, ff_dim, seq_length, embedding_dim, num_layers, vocab_size, optimizer='adam'):
        super(TransformerModel, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.ff_dim = ff_dim
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.optimizer = tf.keras.optimizers.get(optimizer)
        
        self.encoder_embedding = Embedding(vocab_size, embedding_dim)
        self.decoder_embedding = Embedding(vocab_size, embedding_dim)
        
        self.encoder_layers = [TransformerEncoderLayer(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim) for _ in range(self.num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim) for _ in range(self.num_layers)]
        
        self.encoder_positional_encoding = PositionalEncoding(seq_length, embedding_dim)
        self.decoder_positional_encoding = PositionalEncoding(seq_length, embedding_dim)
        self.flatten = Flatten()
        
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, encoder_input, decoder_input, training=True):
        # Encoder
        x = self.encoder_embedding(encoder_input)
        x = self.encoder_positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training)
        encoder_output = x

        # Decoder
        y = self.decoder_embedding(decoder_input)
        y = self.decoder_positional_encoding(y)
        for decoder_layer in self.decoder_layers:
            y = decoder_layer(y, encoder_output, training)
        
        decoder_output = self.dense(self.flatten(y))
        return decoder_output

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        loss_value = loss_obj(real, pred)
        mask = tf.cast(mask, dtype=loss_value.dtype)
        loss_value *= mask
        return tf.reduce_mean(loss_value)

    @tf.function
    def train_step(self, inp, tar_inp):
        with tf.GradientTape() as tape:
            loss = 0
            for t in range(1, tar_inp.shape[1]):
                predictions = self([inp, tar_inp[:, :t]], training=True)
                predicted_id = tf.argmax(predictions, axis=-1)
                loss += self.loss_function(tar_inp[:, t], predicted_id)
            loss /= tar_inp.shape[1]
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
        
    


class PrecisionCallback(Callback):
    def __init__(self, encoder_input_data, decoder_input_data, tokenizer, max_decoder_seq_length, model):
        super().__init__()
        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.model = model
        self.tokenizer = tokenizer
        self.max_decoder_seq_length = max_decoder_seq_length

    def decode_sequence(self, input_seq):
        encoder_input = np.array([input_seq])
        decoder_input = np.array([[self.tokenizer.word_index['starttoken']]])

        decoded_sentence = []
        for _ in range(self.max_decoder_seq_length):
            predictions = self.model([encoder_input, decoder_input], training=False)
            predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]
            decoded_sentence.append(predicted_id)
            if predicted_id == self.tokenizer.word_index['endtoken']:
                break
            decoder_input = np.append(decoder_input, [[predicted_id]], axis=1)
        return decoded_sentence

    def on_epoch_end(self, epoch, logs=None):
        y_pred = []
        y_true = []

        for seq_index in range(len(self.encoder_input_data)):
            input_seq = self.encoder_input_data[seq_index]
            true_seq = self.decoder_input_data[seq_index]

            decoded_sentence = self.decode_sequence(input_seq)
            y_pred.append(decoded_sentence)
            y_true.append(true_seq)

        y_pred_flat = [word for sentence in y_pred for word in sentence if word != 0]
        y_true_flat = [word for sentence in y_true for word in sentence if word != 0]

        precision = precision_score(y_true_flat, y_pred_flat, average='micro', labels=list(self.tokenizer.word_index.values()))
        print(f'Precision at epoch {epoch + 1}: {precision:.4f}')

def main() :
    num_departments = 5
    seq_length = 10
    epochs = 10
    embedding_dim = 64
    vocab_size = 5000

    encoder_input_data = np.random.rand(num_departments, seq_length, embedding_dim)
    decoder_input_data = np.random.rand(num_departments, seq_length, embedding_dim)

    num_heads = 8
    key_dim = embedding_dim
    ff_dim = 128
    num_layers = 2

    transformer_model = TransformerModel(num_heads, key_dim, ff_dim, seq_length, embedding_dim, num_layers, vocab_size, 'adam')

    for epoch in range(epochs):
        total_loss = 0
        for batch, (inp, tar_inp) in enumerate(zip(encoder_input_data, decoder_input_data)):
            batch_loss = transformer_model.train_step((inp, decoder_input_data[batch]), tar_inp)
            total_loss += batch_loss

        print(f'Epoch {epoch+1}, Loss: {total_loss.numpy() / (batch+1)}')
        precision = PrecisionCallback(encoder_input_data, decoder_input_data)
        precision.on_epoch_end(epoch)
        if (epoch + 1) % 2 == 0:
            transformer_model.save('models/Transformer_{epoch:02d}.h5'.format(epoch+1))

if __name__ == '__main__':
    main()
