import numpy as np
import tensorflow as tf
import pandas as pd
from train import AttRNN, DataPreprocessor

# Collect user input
print("Please enter the following values for inference:")

lq_month1_emplvl = float(input("Enter lq_month1_emplvl: "))
lq_month2_emplvl = float(input("Enter lq_month2_emplvl: "))
lq_month3_emplvl = float(input("Enter lq_month3_emplvl: "))

oty_month1_emplvl_pct_chg = float(input("Enter oty_month1_emplvl_pct_chg: "))
oty_month2_emplvl_pct_chg = float(input("Enter oty_month2_emplvl_pct_chg: "))
oty_month3_emplvl_pct_chg = float(input("Enter oty_month3_emplvl_pct_chg: "))

lq_qtrly_contributions = float(input("Enter lq_qtrly_contributions: "))
oty_qtrly_contributions_pct_chg = float(input("Enter oty_qtrly_contributions_pct_chg: "))
lq_avg_wkly_wage = float(input("Enter lq_avg_wkly_wage: "))


user_data = {
    'lq_month1_emplvl': lq_month1_emplvl,
    'lq_month2_emplvl': lq_month2_emplvl,
    'lq_month3_emplvl': lq_month3_emplvl,
    'oty_month1_emplvl_pct_chg': oty_month1_emplvl_pct_chg,
    'oty_month2_emplvl_pct_chg': oty_month2_emplvl_pct_chg,
    'oty_month3_emplvl_pct_chg': oty_month3_emplvl_pct_chg,
    'lq_qtrly_contributions': lq_qtrly_contributions,
    'oty_qtrly_contributions_pct_chg': oty_qtrly_contributions_pct_chg,
    'lq_avg_wkly_wage': lq_avg_wkly_wage,
}

dataframe_path = 'filtered_data.csv'
dataframe = pd.read_csv(dataframe_path)
gru_units = 64
input_df = pd.DataFrame([user_data])
preprocessor = DataPreprocessor(dataframe, batch_size=53)
_, _, size1, size2 = preprocessor.preprocess()


emplvl_combined = np.concatenate([
    [input_df['lq_month1_emplvl'].values[0]],
    [input_df['lq_month2_emplvl'].values[0]],
    [input_df['lq_month3_emplvl'].values[0]]
], axis=None)


pct_chg_combined = np.concatenate([
    [input_df['oty_month1_emplvl_pct_chg'].values[0]],
    [input_df['oty_month2_emplvl_pct_chg'].values[0]],
    [input_df['oty_month3_emplvl_pct_chg'].values[0]]
], axis=None)

def tokenize_and_pad(sequence, max_len=3):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sequence.astype(str))
    tokenized_sequence = tokenizer.texts_to_sequences(sequence.astype(str))
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sequence, maxlen=max_len, padding='post')
    return padded_sequence, len(tokenizer.word_index) + 1

emplvl_combined_padded, emplvl_vocab = tokenize_and_pad(emplvl_combined)
pct_chg_combined_padded, chg_vocab = tokenize_and_pad(pct_chg_combined)


lq_qtrly_contributions_aug_tensor = tf.convert_to_tensor([user_data['lq_qtrly_contributions']], dtype=tf.float32)
oty_qtrly_contributions_pct_chg_aug_tensor = tf.convert_to_tensor([user_data['oty_qtrly_contributions_pct_chg']], dtype=tf.float32)
lq_avg_wkly_wage_tensor = tf.convert_to_tensor([user_data['lq_avg_wkly_wage']], dtype=tf.float32)


inputs_dict = {
    'emplvl_combined': tf.convert_to_tensor(emplvl_combined_padded, dtype=tf.float32),
    'pct_chg_combined': tf.convert_to_tensor(pct_chg_combined_padded, dtype=tf.float32),
    'lq_qtrly_contributions_aug': lq_qtrly_contributions_aug_tensor,
    'oty_qtrly_contributions_pct_chg_aug': oty_qtrly_contributions_pct_chg_aug_tensor,
    'lq_avg_wkly_wage': lq_avg_wkly_wage_tensor
}




dataframe_path = 'filtered_data.csv'
dataframe = pd.read_csv(dataframe_path)
gru_units = 64
input_df = pd.DataFrame([user_data])
preprocessor = DataPreprocessor(dataframe, batch_size=53)
_, _, size1, size2 = preprocessor.preprocess()


attrnn = AttRNN(size1, size2, gru_units)
prediction = attrnn(inputs_dict, training=False)

print(f"Predicted lq_total_qtrly_wages: {prediction.numpy()}")