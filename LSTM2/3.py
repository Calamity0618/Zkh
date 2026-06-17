import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf


class LSTM(object):
    def __init__(self):
        pass

    def create_dataset(self, df, target='count', n_steps=7):
        values = df.values
        target_idx = df.columns.get_loc(target)
        X, y = [], []
        for i in range(len(values) - n_steps):
            X.append(values[i:i+n_steps])
            y.append(values[i+n_steps, target_idx])
        return np.array(X), np.array(y)

    def build_lstm(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(50, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def evaluate(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
if __name__ == '__main__':
    df_base = pd.read_csv('scenic_data_baseline.csv')
    df_enh = pd.read_csv('scenic_data_enhanced.csv')
    
    # 构造数据集
    n_steps = 7
    lstm = LSTM()
    train_size = int(0.8 * len(df_base))
    # 模型A 纯历史
    X_a, y_a = lstm.create_dataset(df_base)
    model_a = lstm.build_lstm(n_steps, X_a.shape[2])
    model_a.fit(X_a[:train_size], y_a[:train_size], epochs=50, validation_data=(X_a[train_size:], y_a[train_size:]), batch_size=8, verbose=2)
    pred_a = model_a.predict(X_a[train_size:], verbose=0).flatten()
    metrics_a = lstm.evaluate(y_a[train_size:], pred_a)
    
    # 模型B 增强特征
    X_b, y_b = lstm.create_dataset(df_enh)
    model_b = lstm.build_lstm(n_steps, X_b.shape[2])
    model_b.fit(X_b[:train_size], y_b[:train_size], epochs=50, validation_data=(X_b[train_size:], y_b[train_size:]), batch_size=8, verbose=2)
    pred_b = model_b.predict(X_b[train_size:], verbose=0).flatten()
    metrics_b = lstm.evaluate(y_b[train_size:], pred_b)

    # 结果比对
    comparison = pd.DataFrame({
        "纯历史（模型A）": metrics_a,
        "增强特征（模型B）": metrics_b
    })
    print(comparison.T.round(4))

    print("\n天气数据带来的提升↓")
    for k in ['mae', 'rmse', 'mape']:
        gain = (metrics_b[k] - metrics_a[k]) / metrics_a[k] * 100
        print(f"{k}: {gain:.2f}%")
