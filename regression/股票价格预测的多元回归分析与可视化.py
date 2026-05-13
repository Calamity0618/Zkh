import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt
import seaborn as sns

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

# 多元回归，假设下一个交易日股价跟成交量、大盘涨跌、大盘成交量、大单特大单成交量有关系
# 获取华夏银行日线数据
df = pd.read_sql_query(
    """
    SELECT d.*, m.buy_lg_vol, m.sell_lg_vol, m.buy_elg_vol, m.sell_elg_vol, m.net_mf_vol, i.vol as i_vol, i.closes as i_closes FROM date_1 d
    left JOIN moneyflows m on d.ts_code = m.ts_code and d.trade_date = m.trade_date
    left join index_daily i on d.trade_date = i.trade_date and i.ts_code = '399001.SZ'
    WHERE d.trade_date BETWEEN '2023-01-01' and '2023-12-31' AND d.ts_code = '002229.SZ'
    """,
    conn,
    chunksize=chunk_size,
)

df1 = pd.concat(df, ignore_index=True)
print(df1.head())

# 计算涨跌幅和成交量变化率
df1['zd_closes'] = round((df1['closes'] - df1['closes'].shift(1)) / df1['closes'].shift(1), 2)
df1['zs_closes'] = round((df1['i_closes'] - df1['i_closes'].shift(1)) / df1['i_closes'].shift(1), 2)
df1['zs_vol'] = round((df1['i_vol'] - df1['i_vol'].shift(1)) / df1['i_vol'].shift(1), 2)

print(df1.head())

# 处理缺失数据
df1 = df1.dropna(subset=['zd_closes', 'zs_closes', 'zs_vol'])
print(df1.head())

# 筛选自变量
ex = ['id', 'ts_code', 'trade_date', 'the_date', 'opens', 'high', 'low', 'closes', 'pre_closes', 'changes',
      'pct_chg', 'amount', 'i_vol', 'i_closes']
number = df1.select_dtypes(include=['number']).columns.to_list()
newList = [col for col in number if col not in ex]

# 回归分析
formula = f"zd_closes ~ {' + '.join(newList)}"
res = smf.ols(formula, data=df1).fit()
print(res.summary())

# 绘制拟合值与实际值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(res.fittedvalues, df1['zd_closes'])
plt.show()

# 特征相关性热力图
features = df1[['vol', 'amount', 'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol', 'net_mf_vol']]
correlation = features.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', cbar=True, fmt='.2f', linewidths=0.5)
plt.show()