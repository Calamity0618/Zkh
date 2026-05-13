import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# 配置连接
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'sjk1234',
    'database': 'tushare',
    'charset': 'utf8mb4',

}
db_engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset={db_config['charset']}")
conn = pymysql.connect(**db_config)
chunk_size = 10000

# 主成分分析
# 获取华夏银行日线数据
df = pd.read_sql_query(
    """
    SELECT d.closes, d.vol, d.amount, m.buy_sm_vol, m.sell_sm_vol, m.sell_lg_vol, m.buy_elg_vol, 
    m.sell_elg_vol, m.net_mf_vol FROM date_1 d
    left JOIN moneyflows m on d.ts_code = m.ts_code and d.trade_date = m.trade_date
    WHERE d.trade_date BETWEEN '2023-01-01' and '2023-12-31' AND d.ts_code = '002229.SZ'
    """,
    conn,
    chunksize=chunk_size,
)

df1 = pd.concat(df, ignore_index=True)
print(df1.head())

df1['zd_closes'] = round((df1['closes'] - df1['closes'].shift(1)) / df1['closes'].shift(1), 2)

# 处理缺失数据
df1 = df1.dropna(subset=['zd_closes']).reset_index(drop=True)
print(df1.head())

numeric_cols = df1.select_dtypes(include=['number']).columns.to_list()

# 选择需要进行主成分分析的自变量
features = df1[['closes', 'vol', 'amount', 'buy_sm_vol', 'sell_sm_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol', 'net_mf_vol']]

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(np.cov(features, rowvar=False))
print("累计共享率为：", round(eigenvalues[:5].sum() / eigenvalues.sum(), 4) * 100, "%")
# 选择要保留的主成分个数
n_components = 5
top_components = eigenvectors[:, :n_components]
print("选择的主成分向量：")

# 计算主成分
principal_components = np.dot(features, top_components)
print("主成分：")
print(principal_components)

# 将主成分添加到原数据中
principal_components = np.dot(features, top_components)
data_pca = pd.concat([df1, pd.DataFrame(principal_components, columns=[f'PC{i}' for i in range(n_components)])], axis=1)
print(data_pca)

# 添加常数列前确保数据类型正确
X_pca = data_pca[[f'PC{i + 1}' for i in range(n_components)]].copy()
X_pca = sm.add_constant(X_pca)

# 确保y的索引与X_pca一致
y = data_pca['zd_closes'].copy()

# 构建回归模型
model_pca = sm.OLS(y, X_pca)
# 拟合模型
result_pca = model_pca.fit()
# 输出结果
print(result_pca.summary())

# 选择PC1，PC2，PC3作为新的自变量
X_pca_selected = data_pca[['PC1', 'PC2', 'PC3']]
X_pca_selected.columns = ['PC1', 'PC2', 'PC3']

# 添加常数列
X_pca_selected = sm.add_constant(X_pca_selected)
# 因变量
y = data_pca['zd_closes'].copy()
model_pca_selected = sm.OLS(y, X_pca_selected)
# 拟合模型
result_pca_selected = model_pca_selected.fit()
# 输出结果
print(result_pca_selected.summary())

# 绘制散点图
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for i, col in enumerate(X_pca_selected.columns):
    axes[i].scatter(X_pca_selected[col], y, s=50, alpha=0.7)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('zd_closes')
    axes[i].set_title(f'PC{i + 1} vs zd_closes')

plt.tight_layout()
plt.show()

# 计算主成分构成
for k in range(0, 5):
    string_y = f'PC{k + 1} = '
    i = eigenvectors[k]
    for j in range(len(i)):
        if i[j] > 0:
            string_y = string_y + f'+{round(i[j], 2)} * X_{j + 1}'
        else:
            string_y = string_y + f'{round(i[j], 2)} * X_{j + 1}'
    if k != 2 and k != 4:
        print(string_y)
    