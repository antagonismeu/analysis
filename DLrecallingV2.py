import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


multiplier = 1
model = load_model(f'models/DL{multiplier}x.h5')





def detransform_log(y):
    return y / multiplier



def get_input_data():
    print("Please enter the following inputs:")
    lq_month1_emplvl = float(input("lq_month1_emplvl: "))
    oty_month1_emplvl_pct_chg = float(input("oty_month1_emplvl_pct_chg: "))
    lq_month2_emplvl = float(input("lq_month2_emplvl: "))
    oty_month2_emplvl_pct_chg = float(input("oty_month2_emplvl_pct_chg: "))
    lq_month3_emplvl = float(input("lq_month3_emplvl: "))
    oty_month3_emplvl_pct_chg = float(input("oty_month3_emplvl_pct_chg: "))
    lq_qtrly_contributions = float(input("lq_qtrly_contributions: "))
    oty_qtrly_contributions_pct_chg = float(input("oty_qtrly_contributions_pct_chg: "))
    lq_avg_wkly_wage = float(input("lq_avg_wkly_wage: "))

    
    return np.array([[lq_month1_emplvl, oty_month1_emplvl_pct_chg, lq_month2_emplvl, 
                      oty_month2_emplvl_pct_chg, lq_month3_emplvl, oty_month3_emplvl_pct_chg,
                      lq_qtrly_contributions, oty_qtrly_contributions_pct_chg, lq_avg_wkly_wage]]) * multiplier


def normalize_input(input_data, normalizer):
    return normalizer(input_data)



while True:
    user_input = input("Continue with new input? (Press 'q' to quit, any other key to continue): ")
    
    if user_input.lower() == 'q':
        print("Exiting the program.")
        break
    input_data = get_input_data()



    intermediate = model.predict(input_data * multiplier)


    print(f"Predicted lq_total_qtrly_wages: {detransform_log(intermediate)[0][0]:.4f}")


