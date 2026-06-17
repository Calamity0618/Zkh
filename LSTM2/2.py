# 导入库
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
    df = pd.read_csv('scenic_data_baseline.csv')
    
    # 构造数据集
    lstm = LSTM()
    X, y = lstm.create_dataset(df)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 训练基准模型
    model = lstm.build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=8, verbose=2)

    # 评估
    y_pred = model.predict(X_test, verbose=0).flatten()
    metrics = lstm.evaluate(y_test, y_pred)
    print(f'MAE: {metrics["mae"]:.4f}')
    print(f'RMSE: {metrics["rmse"]:.4f}')
    print(f'MAPE: {metrics["mape"]:.4f}%')

    # 保存模型
    model.save('lstm_model.keras')
