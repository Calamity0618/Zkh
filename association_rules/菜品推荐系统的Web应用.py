# 导入库
from flask import Flask, request, jsonify, render_template
import pandas as pd
app = Flask(__name__)
# 加载模型
frequent_itemsets = pd.read_pickle('./association_rules/frequent_itemsets.pkl')
rules = pd.read_pickle('./association_rules/rules.pkl')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json.get('items', [])
    # 生成关联规则推荐
    recommendDations = []
    for idx, rule in rules.iterrows():
        antecedents = list(rule['antecedents'])
        consequents = list(rule['consequents'])
        # 检查前件是否跟输入匹配
        if set(antecedents).issubset(set(data)):
            recommendDations.extend(consequents)

    # 去重
    recommendDations = list(set(recommendDations) - set(data))
    
    # 当推荐内容为空时，返回一个热门菜品
    if not recommendDations:
        # 取最频繁的第一项，转成列表返回
        top_item = frequent_itemsets.sort_values('support', ascending=False).iloc[0]
        recommendDations = list(top_item['itemsets'])

    # 返回推荐结果
    return jsonify({'recommendDations': recommendDations})

if __name__ == "__main__":
    app.run(debug=True)
