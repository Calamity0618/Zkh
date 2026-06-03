# 导入库
import pandas as pd
import numpy as np
import pymysql
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

class MysqlUtils(object):
    def __init__(self):
        self.conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='sjk1234',
            database='scenic_data',
            charset='utf8mb4',
        )

    def is_holiday(self, date):
        if date in ['2024-09-03', '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06', '2024-10-07', '2025-01-01']:
            return 1
        return 0

    def get_data(self):
        cursor = self.conn.cursor(pymysql.cursors.DictCursor)
        sql = """
        SELECT DATE(gate.create_time) as date, HOUR(gate.create_time) as hour, count(*) as count
        FROM order_user_gate_rel gate WHERE HOUR(gate.create_time) BETWEEN 6 and 23 GROUP BY date, hour
        """
        cursor.execute(sql)
        ret = cursor.fetchall()
        df = pd.DataFrame(ret)
        # print(df.head())
        # 格式转换
        date_range = pd.date_range(start='2024-07-01', end='2025-01-01', freq='D')
        hours = range(6, 24)
        full_index = pd.MultiIndex.from_product([date_range, hours], names=['date', 'hour'])
        df_full = df.set_index(['date', 'hour']).reindex(full_index, fill_value=0).reset_index()
        # 按天组织数据，每行包含18个小时段的检票次数
        df_pivot = df_full.pivot(index='date', columns='hour', values='count')
        # print(df_pivot.head())
        # 周几、月份、是否节假日
        df_pivot['dow'] = df_pivot.index.dayofweek  # 星期几（0-6）
        df_pivot['month'] = df_pivot.index.month    # 月份（1-12）
        df_pivot['is_holiday'] = df_pivot.index.map(self.is_holiday)
        # print(df_pivot.head())

        # 对星期几和月份进行独热编码
        df_pivot = pd.get_dummies(df_pivot, columns=['dow', 'month'], dtype=int)
        print(df_pivot.head())
        
        # 归一化小时列
        hours_columns = list(range(6, 24))
        df_hours = df_pivot[hours_columns].copy()

        feature_columns = [col for col in df_pivot.columns if col not in hours_columns]
        df_features = df_pivot[feature_columns].copy()

        scaler = MinMaxScaler()
        scaler_hours = scaler.fit_transform(df_hours)
        dump(scaler, 'scaler.joblib')

        # 将归一化后的数据转换为DataFrame
        df_hours_scaled = pd.DataFrame(scaler_hours, columns=hours_columns, index=df_features.index)
        # 合并
        df_pivot_clean = pd.concat([df_hours_scaled, df_features], axis=1)
        print(df_pivot_clean.head())

        # 保存数据
        df_pivot_clean.to_csv('df_pivot_clean.csv', index=False)

if __name__ == '__main__':
    mu = MysqlUtils()
    mu.get_data()