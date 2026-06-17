# 消融实验：逐个剔除特征，查看贡献度
import pandas as pd
import numpy as np
import tensorflow as tf

class AblationStudy(object):
    def __init__(self):
        self.df = pd.read_csv('df_pivot_weather.csv')
        self.label_column = 'total_count_scaled'
        self.feature_columns = [col for col in self.df.columns if col != self.label_column and col != 'date' and col != 'total_count']
        
        # 天气相关特征
        self.weather_features = ['temperature', 'humidity', 'wind_speed', 'weather_code']
    
    def create_dataset(self, features, labels, n_steps=7):
        """创建时间序列数据集"""
        X, y = [], []
        for i in range(len(features) - n_steps):
            X.append(features[i:i+n_steps])
            y.append(labels[i+n_steps])
        return np.array(X), np.array(y)
    
    def train_and_evaluate(self, features, labels, n_steps=7):
        """训练模型并返回测试损失"""
        X, y = self.create_dataset(features, labels, n_steps)
        
        # 划分训练集和测试集
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 构建模型
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(n_steps, X_train.shape[2])),
            tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
            tf.keras.layers.LSTM(50, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        
        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss
    
    def run_ablation(self):
        """运行消融实验"""
        print("=== 消融实验开始 ===\n")
        
        # 完整模型（所有特征）
        features_full = self.df[self.feature_columns].values
        labels = self.df[self.label_column].values
        full_loss = self.train_and_evaluate(features_full, labels)
        print(f"完整模型（含所有天气特征）测试损失: {full_loss:.4f}")
        print(f"特征数量: {len(self.feature_columns)}\n")
        
        # 逐个剔除天气特征
        results = {}
        for feature in self.weather_features:
            if feature in self.feature_columns:
                # 剔除当前特征
                ablation_features = [f for f in self.feature_columns if f != feature]
                features_ablation = self.df[ablation_features].values
                
                loss = self.train_and_evaluate(features_ablation, labels)
                impact = ((loss - full_loss) / full_loss) * 100 if full_loss > 0 else 0
                results[feature] = {'loss': loss, 'impact': impact}
                print(f"剔除 {feature} 后:")
                print(f"  测试损失: {loss:.4f}")
                print(f"  性能下降: {impact:.2f}%\n")
        
        # 剔除所有天气特征
        non_weather_features = [f for f in self.feature_columns if f not in self.weather_features]
        features_no_weather = self.df[non_weather_features].values
        no_weather_loss = self.train_and_evaluate(features_no_weather, labels)
        impact = ((no_weather_loss - full_loss) / full_loss) * 100 if full_loss > 0 else 0
        results['all_weather'] = {'loss': no_weather_loss, 'impact': impact}
        print(f"剔除所有天气特征后:")
        print(f"  测试损失: {no_weather_loss:.4f}")
        print(f"  性能下降: {impact:.2f}%\n")
        
        # 输出特征贡献度排序
        print("\n=== 特征贡献度分析 ===")
        print(f"{'特征':<20} {'剔除后损失':<15} {'贡献度(%)':<15}")
        print("-" * 50)
        
        sorted_features = sorted(results.items(), key=lambda x: x[1]['impact'], reverse=True)
        for feature, metrics in sorted_features:
            print(f"{feature:<20} {metrics['loss']:<15.4f} {metrics['impact']:<15.2f}")
        
        print("\n结论:")
        if sorted_features:
            most_important = sorted_features[0]
            print(f"最重要的天气特征: {most_important[0]} (贡献度: {most_important[1]['impact']:.2f}%)")

if __name__ == '__main__':
    ablation = AblationStudy()
    ablation.run_ablation()
