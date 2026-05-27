import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pymysql


# ==================== 数据库工具类 ====================
class MysqlUtils(object):
    def __init__(self, host, port, user, password, database):
        self.conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )

    def get_data(self, sql):
        cursor = self.conn.cursor(cursor=pymysql.cursors.DictCursor)
        cursor.execute(sql)
        ret = cursor.fetchall()
        df = pd.DataFrame(ret)
        cursor.close()
        return df

    def close(self):
        self.conn.close()


# ==================== KMeans 聚类优化类 ====================
class KMeansOptimization(object):
    def __init__(self, features):
        self.features = features

    def elbow_method(self):
        """肘部法则确定最佳K值"""
        k_range = range(1, 11)
        sse = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(10, 5))
        plt.plot(list(k_range), sse, 'bo-', linewidth=2, markersize=8)
        plt.ylabel('SSE')
        plt.title('肘部法则')
        plt.grid(True, linestyle='--', alpha=0.5)

        # 标注每个点的SSE值
        for i, sse_val in enumerate(sse):
            plt.annotate(
                f'{sse_val:.2f}',
                (list(k_range)[i], sse_val),
                textcoords='offset points',
                xytext=(0, 10),
                fontsize=10,
                ha='center'
            )

        plt.tight_layout()
        plt.savefig('elbow_method.png', dpi=300)
        plt.show()

    def silhouette_analysis(self):
        """轮廓系数分析，选择轮廓系数最高的K值"""
        k_range = range(2, 10)
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.features)
            silhouette_scores.append(silhouette_score(self.features, kmeans.labels_))
            print(f'K={k}, 轮廓系数: {silhouette_scores[-1]:.4f}')

        # 轮廓系数图
        plt.figure(figsize=(10, 5))
        plt.plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
        plt.xlabel('K值')
        plt.ylabel('轮廓系数')
        plt.title('轮廓系数分析')
        plt.xticks(list(k_range))
        plt.grid(True, linestyle='--', alpha=0.5)

        best_k_sil = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
        plt.axvline(best_k_sil, color='red', linestyle='--', label=f'最佳 K={best_k_sil}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('silhouette_analysis.png', dpi=300)
        plt.show()

        print(f'最佳 K值: {best_k_sil}')
        return best_k_sil


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # 连接数据库
    db = MysqlUtils(
        host='localhost',
        port=3306,
        user='root',
        password='sjk1234',
        database='scenic_data'
    )

    # 查询数据
    sql = """
    SELECT t.tourist_agency_name, rel.id_no, LEFT(rel.id_no, 2) as province_code,
    CAST(SUBSTRING(rel.id_no, 7, 4) AS UNSIGNED) as birth_year,
    DAYOFWEEK(gate.create_time) as weekend
    FROM ticket_order_user_rel rel
    JOIN ticket_order t ON t.id = rel.order_id
    JOIN order_user_gate_rel gate ON gate.ticket_rel_id = rel.id
    WHERE t.tourist_agency_name != '' AND t.pay_time IS NOT NULL
    """
    df = db.get_data(sql)

    # 数据预处理
    df['non_weekend'] = df['weekend'].apply(lambda x: 1 if x in [1, 7] else 0)
    df['valid_id'] = df['id_no'].apply(lambda x: 1 if x and str(x).strip() != '' else 0)
    df['elderly'] = df.apply(
        lambda x: 1 if x['valid_id'] and 2026 - x['birth_year'] >= 60
        else 0 if x['valid_id'] else np.nan, axis=1
    )
    df['outProvince'] = df.apply(
        lambda x: 1 if x['valid_id'] and x['province_code'] != '44'
        else 0 if x['valid_id'] else np.nan, axis=1
    )

    # 按旅行社分组聚合
    result = df.groupby('tourist_agency_name').agg(
        total_visitors=('id_no', 'count'),
        valid_visitors=('valid_id', 'sum'),
        outProvince_visitors=('outProvince', 'sum'),
        elderly_visitors=('elderly', 'sum'),
        non_weekend_ratio=('non_weekend', 'mean')
    ).reset_index()

    result['outProvince_ratio'] = result['outProvince_visitors'] / result['total_visitors'].replace(0, np.nan)
    result['elderly_ratio'] = result['elderly_visitors'] / result['total_visitors'].replace(0, np.nan)
    result = result.drop(['outProvince_visitors', 'elderly_visitors'], axis=1)
    result['outProvince_ratio'] = result['outProvince_ratio'].fillna(0)
    result['elderly_ratio'] = result['elderly_ratio'].fillna(0)

    # 提取聚类特征
    features = result[['non_weekend_ratio', 'outProvince_ratio', 'elderly_ratio']].values

    # 聚类优化分析
    kmeans_opt = KMeansOptimization(features)
    kmeans_opt.elbow_method()
    best_k = kmeans_opt.silhouette_analysis()
    print(f'最终选定的最佳K值: {best_k}')

    # 关闭数据库连接
    db.close()
