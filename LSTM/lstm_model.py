# 导入库
import pandas as pd
import numpy as np
import tensorflow as tf


class LSTM(object):
    def __init__(self):
        self.df = pd.read_csv('df_pivot_clean.csv')

    def create_dataset(self, data, n_steps=7):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps])
            y.append(data[i+n_steps, :18])
        return np.array(X), np.array(y)

    def get_model(self):
        n_steps = 7  # 步长7天
        data = self.df.values
        X, y = self.create_dataset(data, n_steps)

        # 划分训练集和测试集
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 构建模型
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(20, activation='relu', return_sequences=True, input_shape=(n_steps, X_train.shape[2])))
        model.add(tf.keras.layers.LSTM(20, activation='relu'))
        model.add(tf.keras.layers.Dense(18))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        # 评估
        loss = model.evaluate(X_test, y_test)
        print(f"测试集损失: {loss:.4f}")

        # 保存模型
        tf.keras.models.save_model(model, 'lstm_model.keras')


if __name__ == '__main__':
    lstm = LSTM()
    lstm.get_model()
