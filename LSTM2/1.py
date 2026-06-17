# 导入库
import datetime

import pandas as pd
import numpy as np
import pymysql
import requests
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

class MysqlUtils(object):
    def __init__(self):
        self.conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='sjk1234',
            database='scenic',
            charset='utf8mb4',
            ssl_disabled=True
        )

    def is_holiday(self, date):
            # 指定节假日日期列表
            holiday_list = ['2024-09-03', '2024-10-01', '2024-10-02', '2024-10-03', '2024-10-04', '2024-10-05', '2024-10-06']
            if date in holiday_list:
                return 1
            return 0
        
    def get_data(self):
        cursor = self.conn.cursor(pymysql.cursors.DictCursor)
        sql = """
        SELECT DATE(gate.create_time) as date, count(*) as count
FROM order_user_gate_rel gate 
GROUP BY date
        """
        cursor.execute(sql)
        ret = cursor.fetchall()
        df = pd.DataFrame(ret)
        return df
    
class WeatherUtils(object):
    def __init__(self):
        self.api = 'http://v1.yiketianqi.com/api'
        self.appid = '75757364' # 替换成自己的appid
        self.secret = 'lyo1U3Cu' # 替换成自己的secret
        self.date_list = [
            '2024-07-01',
            '2024-08-01',
            '2024-09-01',
            '2024-10-01',
            '2024-11-01',
            '2024-12-01',
            '2025-01-01',
        ]
    def get_weather(self):
        data_list = []
        for date in self.date_list:
            conf = {
                "appid": self.appid,
                "appsecret": self.secret,
                "version": "history",
                "year": date.split('-')[0],
                "month": date.split('-')[1],
                "city": "南昌"
        }
        # 发起请求获取数据
        response = requests.get(self.api + '?', params=conf)
        res_data = response.json()
        for i in res_data['data']:
                data_list.append({
                    "date": datetime.datetime.strptime(i['ymd'], '%Y-%m-%d'),
                    "bWendu": i['bWendu'],
                    "yWendu": i['yWendu'],
                    "tianqi": i['tianqi'],
                    "fengli": i['fengli'],
                })
        df = pd.DataFrame(data_list)
        df.to_csv('weather.csv', index=False)
            
        def get_build_features(self, df):
            df['dow'] = df.index.dayofweek  # 星期几（0-6）
            df['month'] = df.index.month    # 月份（1-12）
            df['is_holiday'] = df.index.map(self.is_holiday)

            # 温度数值化
            df['bWendu'] = df['bWendu'].str.replace('°', '').astype(int)
            df['yWendu'] = df['yWendu'].str.replace('°', '').astype(int)

            # 周几、月份、天气、风力独热编码
            df = pd.get_dummies(df, columns=['dow', 'month', 'tianqi', 'fengli'])
            return df
        

if __name__ == '__main__':
    mu = MysqlUtils()
    df_scenic = mu.get_data()
    weather_utils = WeatherUtils()
    # weather_utils.get_weather()
    df_weather = pd.read_csv('weather.csv')

    df_weather['date'] = pd.to_datetime(df_weather['date'])
    df_scenic['date'] = pd.to_datetime(df_scenic['date'])

    # 按日期合并
    df = pd.merge(df_scenic, df_weather, on='date', how='inner').set_index('date')
    
    #分离目标和特征
    # 分离目标和特征
    baseline_features = [c for c in df.columns if c not in ['count', 'bWendu', 'yWendu', 'tianqi', 'fengli'] and not c.startswith('tianqi_') and not c.startswith('fengli_')]
    enhanced_features = [c for c in df.columns if c not in ['count', 'tianqi', 'fengli']]

    df_baseline = df[['count'] + baseline_features]
    df_enhanced = df[['count'] + enhanced_features]

    # 归一化并保存
    scaler_base = MinMaxScaler()
    scaler_enh = MinMaxScaler()

    pd.DataFrame(scaler_base.fit_transform(df_baseline), columns=df_baseline.columns).to_csv('df_baseline.csv', index=False)
    pd.DataFrame(scaler_enh.fit_transform(df_enhanced), columns=df_enhanced.columns).to_csv('df_enhanced.csv', index=False)