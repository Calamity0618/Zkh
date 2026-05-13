# 导入库
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 读取数据
df = pd.read_excel('./association_rules/餐厅数据.xlsx')
print(df.head())

# 标准化
trsations = df['菜品'].str.split(',').to_list()

te = TransactionEncoder()
te_ary = te.fit(trsations).transform(trsations)
df_enecoded = pd.DataFrame(te_ary, columns=te.columns_)
print(df_enecoded)

# 使用apriori进行分析
frequent_itemsets = apriori(df_enecoded, min_support=0.1, use_colnames=True)

# 生成关联规则
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=0.1)
rules = rules.sort_values(by=['lift'], ascending=False)

# 保存模型
frequent_itemsets.to_pickle('./association_rules/frequent_itemsets.pkl')
rules.to_pickle('./association_rules/rules.pkl')