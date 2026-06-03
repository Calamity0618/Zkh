import pandas as pd
import pymysql
import pymysql.cursors

class MysqlUtils(object):
    def __init__(self):
        self.conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password="sjk1234",
            database="tushare",
            charset="utf8mb4"
        )

class Classification(object):
    def __init__(self):
        pass

    # 获取财务指标数据
    def get_fina_indicator(self, conn):
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = """
        SELECT ts_code, ann_date, eps,total_revenue_ps, undist_profit_ps, gross_margin, fcff, fcfe, tangible_asset, bps,netprofit_margin, npta
        FROM financial_data WHERE ann_date >= '2023-01-01' and ann_date < '2024-01-01'
        """
        cursor.execute(sql)
        df = pd.DataFrame(cursor.fetchall())
        print(df)
        # 处理缺失值
        df = df.dropna(subset=['eps', 'total_revenue_ps', 'undist_profit_ps', 'gross_margin', 'fcff', 'fcfe', 'tangible_asset', 'bps', 'netprofit_margin', 'npta'])
        # 重建索引
        df = df.reset_index(drop=False)
        print(df)
        return df

    def get_daily(self, conn, df):
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        new_list = []
        for i in range(len(df['ts_code'])):
            print(i)
            ann_date = df['ann_date'][i].strftime('%Y-%m-%d')
            sql = "select trade_date, closes from date_1 where ts_code = '{}' and trade_date > Date('{}') order by trade_date asc limit 20".format(df['ts_code'][i], ann_date)
            cursor.execute(sql)
            df1 = pd.DataFrame(cursor.fetchall())
            try:
                if len(df1) > 0:
                    max_close = df1['closes'].max()
                    min_close = df1['closes'].min()
                    the_close = df1['closes'].iloc[1]
                    new_list.append({
                        'ts_code': df['ts_code'][i],
                        'ann_date': ann_date,
                        'max_close': max_close,
                        'min_close': min_close,
                        'the_close': the_close,
                        'eps': df['eps'][i],
                        'total_revenue_ps': df['total_revenue_ps'][i],
                        'undist_profit_ps': df['undist_profit_ps'][i],
                        'gross_margin': df['gross_margin'][i],
                        'fcff': df['fcff'][i],
                        'fcfe': df['fcfe'][i],
                        'tangible_asset': df['tangible_asset'][i],
                        'bps': df['bps'][i],
                        'netprofit_margin': df['netprofit_margin'][i],
                        'npta': df['npta'][i],
                    })
            except Exception as e:
                print(e)
        df2 = pd.DataFrame(new_list)
        df2.to_csv('date_1.csv', index=False)

if __name__ == '__main__':
    mu = MysqlUtils()
    classification = Classification()
    df = classification.get_fina_indicator(mu.conn)
    classification.get_daily(mu.conn, df)