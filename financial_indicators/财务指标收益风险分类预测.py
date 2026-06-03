# 导入库
import pandas as pd
import numpy as np
from joblib import load

class Classification(object):
    def __init__(self):
        self.knn_model = load('knn_model.joblib')
        self.le = load('le.joblib')
        self.scaler = load('scaler.joblib')

    def get_predictor_conditions(self, X):
        """
        获取分类结果
        """
        # 转换为数组并保持特征顺序
        feature_order = [
            'eps', 'total_revenue_ps', 'undist_profit_ps', 'gross_margin', 'fcff', 'fcfe', 'tangible_asset', 'bps', 'netprofit_margin', 'npta'
        ]
        new_values = np.array([[X[col] for col in feature_order]])
        # 标准化新数据
        new_values_scaled = self.scaler.transform(new_values)
        # 预测新数据
        y_pred = self.knn_model.predict(new_values_scaled)
        # 预测结果转换为标签
        y_pred_labels = self.le.inverse_transform(y_pred)
        return y_pred_labels[0]


if __name__ == '__main__':
    ci = Classification()
    # 000002.SZ 2024-10-31  -1.5132 18.431  6.3427  20894700000  -25189400000   -23392500000   171059000000   19.6233  -7.4573  -1.1459
    new_data = {
        "eps": -1.5132,
        "total_revenue_ps": 18.431,
        "undist_profit_ps": 6.3427,
        "gross_margin": 20894700000,
        "fcff": -25189400000,
        "fcfe": -23392500000,
        "tangible_asset": 171059000000,
        "bps": 19.6233,
        "netprofit_margin": -7.4573,
        "npta": -1.1459
    }
    # 获取分类结果
    predicter = ci.get_predictor_conditions(new_data)
    print(predicter)
