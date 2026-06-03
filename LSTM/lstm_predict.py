import pandas as pd
import numpy as np
import tensorflow as tf
from joblib import load


class LSTM(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('lstm_model.keras')
        self.scaler = load('scaler.joblib')

    def get_hourly_trend(self):
        n_steps = 7  # 步长7天
        df_pivot = pd.read_csv('df_pivot_clean.csv')
        lastest_data = df_pivot[-n_steps:].values  # 最后七天的数据
        lastest_data = lastest_data.reshape(1, n_steps, lastest_data.shape[1])

        # 预测和反归一化
        predicted = self.model.predict(lastest_data)
        predicted_count = self.scaler.inverse_transform(predicted)  # 反归一化
        predicted_count[predicted_count < 0] = 0  # 修正负数
        hourly_trend = predicted_count[0].astype(int).tolist()
        print(hourly_trend)


if __name__ == '__main__':
    lstm = LSTM()
    lstm.get_hourly_trend()
