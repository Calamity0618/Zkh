import pandas as pd
import numpy as np
import pymysql

class MysqlUtils(object):
    def __init__(self):
        self.conn = pymysql.connect(
            host='localhost',
            port=3306,
            user='root',
            password='sjk1234',
            database='scenic_data'
        )
    
    def get_data(self):
        cursor = self.conn.cursor(cursor=pymysql.cursors.DictCursor)
        sql = """
        SELECT t.tourist_agency_name, rel.id_no, LEFT(rel.id_no, 2) as province_code,
        cast(substring(rel.id_no, 7, 4) as unsigned) as birth_year,
        DAYOFWEEK(gate.create_time) as weekend
        FROM ticket_order_user_rel rel
        JOIN ticket_order t on t.id = rel.order_id
        JOIN order_user_gate_rel gate on gate.ticket_rel_id = rel.id
        WHERE t.tourist_agency_name != '' and t.pay_time is not null
        """
        cursor.execute(sql)
        ret = cursor.fetchall()
        df = pd.DataFrame(ret)
        
        # 数据转换
        df['non_weekend'] = df['weekend'].apply(lambda x: 1 if x in [1, 7] else 0)  # 非周末
        
        # 新增有效性标记列
        df['valid_id'] = df['id_no'].apply(lambda x: 1 if x and str(x).strip() != '' else 0)
        
        # 老年人及外省游客
        df['elderly'] = df.apply(
            lambda x: 1 if (x['valid_id'] and 2026 - x['birth_year'] >= 60) else 0 if x['valid_id'] else np.nan,
            axis=1
        )
        df['out_province'] = df.apply(
            lambda x: 1 if (x['valid_id'] and x['province_code'] != '44') else 0 if x['valid_id'] else np.nan,
            axis=1
        )
        
        print(df.head())
        
        # 分组聚合计算占比
        result = df.groupby('tourist_agency_name').agg(
            total_visitors=('id_no', 'count'),  # 总游客数
            valid_visitors=('valid_id', 'sum'),  # 有效游客数
            out_province_visitors=('out_province', 'sum'),  # 外省游客数
            elderly_visitors=('elderly', 'sum'),  # 老年人游客数
            non_weekend_ratio=('non_weekend', 'mean')  # 非周末占比
        ).reset_index()
        
        # 计算实际比例
        result['out_province_ratio'] = result['out_province_visitors'] / result['total_visitors'].replace(0, np.nan)
        result['elderly_ratio'] = result['elderly_visitors'] / result['total_visitors'].replace(0, np.nan)
        
        # 清除中间列
        result = result.drop(['out_province_visitors', 'elderly_visitors'], axis=1)
        result['elderly_ratio'] = result['elderly_ratio'].fillna(0)
        
        print(result.head())
        result.to_csv('tourist_agency_ratio.csv', encoding='utf-8-sig')
        
        return result


# 使用示例
if __name__ == "__main__":
    utils = MysqlUtils()
    result_df = utils.get_data()
    utils.conn.close()
    print("\n结果已保存到 tourist_agency_ratio.csv")