import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from joblib import dump
from sklearn.metrics import accuracy_score


class Classification(object):

    def __init__(self):
        self.df = pd.read_csv('date_1.csv')

    def get_conditions(self):
        """
        分类前准备
        """
        # 计算收益和风险比例
        df = self.df
        df['max_ratio'] = df['max_close'] / df['the_close']  # 收益潜力
        df['min_ratio'] = df['min_close'] / df['the_close']  # 风险程度

        # 自动划分高低阈值
        high_return_threshold = df['max_ratio'].quantile(0.4)  # 前40%定义为高收益
        high_risk_threshold = df['min_ratio'].quantile(0.4)     # 后40%定义为高风险

        # 生成分类标签
        conditions = [
            (df['max_ratio'] >= high_return_threshold) & (df['min_ratio'] <= high_risk_threshold),
            (df['max_ratio'] < high_return_threshold) & (df['min_ratio'] > high_risk_threshold),
            (df['max_ratio'] < high_return_threshold) & (df['min_ratio'] <= high_risk_threshold),
            (df['max_ratio'] >= high_return_threshold) & (df['min_ratio'] > high_risk_threshold)
        ]
        labels = ['高收益高风险', '低收益低风险', '低收益高风险', '高收益低风险']
        df['label'] = np.select(conditions, labels, default='未知')
        print(df['label'].value_counts())
        # 标签选择
        features = df[['eps', 'total_revenue_ps', 'undist_profit_ps', 'gross_margin', 'fcff', 'fcfe', 'tangible_asset', 'bps', 'netprofit_margin', 'npta']]
        # 标签编码
        le = LabelEncoder()
        df['catrgory_encoded'] = le.fit_transform(df['label'])

        # 数据标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        y = df['catrgory_encoded']
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test, le

    def knn_utils(self, X_train, X_test, y_train, y_test, le):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        # 预测与评估
        y_pred = knn.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        # 保存模型
        dump(knn, 'knn_model.joblib')
        dump(le, 'le.joblib')


if __name__ == '__main__':
    ci = Classification()
    X_train, X_test, y_train, y_test, le = ci.get_conditions()
    ci.knn_utils(X_train, X_test, y_train, y_test, le)
