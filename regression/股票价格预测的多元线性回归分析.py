from ast import excepthandler
import pandas as pd
import statsmodels.formula.api as smf
from sqlalchemy import create_engine
import pymysql

# 配置连接
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'sjk1234',
    'database': 'tushare',
    'charset': 'utf8mb4',
}

engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset={db_config['charset']}")
conn = pymysql.connect(**db_config)
chunk_size = 10000

# 一元回归，假设下一个交易日股价跟当前成交量有关系
# 获取华夏银行日线数据
df = pd.read_sql_query(
    """
    SELECT d.* FROM date_1 d WHERE d.trade_date BETWEEN '2023-01-01' and '2023-12-31' AND d.ts_code = '000001.SZ'
    """,
    conn,
    chunksize=chunk_size,
)

df1 = pd.concat(df, ignore_index=True)
print(df1.head())
df1['2d_close'] = round((df1['closes'] - df1['closes'].shift(1) / df1['closes'].shift(1)), 2)
print(df1.head())

# 处理缺失数据
df1 = df1.dropna(subset=["zd_closes"])
print(df1.head())

# 选择需要的列
df1 = df1[['code', 'trade_date', 'the_date', 'opens', 'high', 'low', 'closes', 'pre_closes', 'changes', 'pct_chg', 'amount']]

# 筛选数值型列
number = df1.select_dtypes(include=["number"]).columns.to_list()
newList = [col for col in number if col not in excepthandler]

# 回归分析
formula = f"zd_closes ~ {' + '.join(newList)}"
res = smf.ols(formula, data=df1).fit()
print(res.summary())