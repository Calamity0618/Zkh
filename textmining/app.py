#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask Web应用：LDA主题挖掘 + 情感预测
功能：主题词云展示、主题详情、评论情感预测
作者：AI助手
日期：2026-06-17
学号：23053080649
"""

import os
import re
import base64
import io
import numpy as np
import pandas as pd
import jieba
import joblib
from flask import Flask, render_template, request
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

# ============ Flask 应用初始化 ============
app = Flask(__name__)

# ============ 路径配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'waimai_10k.csv')
FEATURE_DIR = os.path.join(BASE_DIR, 'tfidf_features')

# ============ 停用词表 ============
STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
    '么', '那', '里', '把', '让', '被', '从', '但', '还', '可以',
    '又', '对', '它', '能', '而且', '因为', '所以', '如果', '虽然', '但是',
    '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '哪里', '谁', '何时',
    '个', '只', '条', '张', '本', '把', '件', '样', '种', '些',
    '没', '太', '真', '挺', '比较', '非常', '特别', '就是', '不是',
])


def preprocess_text(text):
    """文本预处理：分词 + 过滤停用词 + 过滤单字词"""
    try:
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(text))
        words = jieba.cut(text)
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 1 and w.strip()]
        return ' '.join(filtered)
    except Exception:
        return ''


def generate_wordcloud_base64(word_freq, color='steelblue'):
    """生成词云图并返回base64编码（文字大小差异化，横竖混合排列）"""
    try:
        # 中文字体路径
        font_path = 'C:/Windows/Fonts/simhei.ttf'

        wc = WordCloud(
            font_path=font_path,
            width=700,
            height=400,
            background_color='white',
            max_words=60,
            max_font_size=160,
            min_font_size=14,
            colormap=color,
            prefer_horizontal=0.6,      # 60%水平，40%竖直
            relative_scaling=0.5,       # 字体大小差异化程度
            random_state=42,
            margin=8,
        )
        wc.generate_from_frequencies(word_freq)

        # 转为base64
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"词云生成失败: {e}")
        return None


def summarize_topic(keywords):
    """根据关键词自动总结主题含义"""
    kw_set = set(keywords)

    # 美食口味类
    taste_words = {'好吃', '味道', '口味', '难吃', '好吃', '鲜', '香', '辣', '甜', '咸', '淡', '油腻', '清淡'}
    # 配送速度类
    speed_words = {'送餐', '速度', '小时', '太慢', '很快', '配送', '送来', '送到', '及时', '快递', '快', '慢', '分钟'}
    # 服务态度类
    service_words = {'态度', '服务', '热情', '客气', '礼貌', '客服', '打电话', '接电话', '回复'}
    # 性价比类
    price_words = {'便宜', '贵', '价格', '划算', '性价比', '实惠', '值得', '分量', '量大', '量少'}
    # 包装品质类
    quality_words = {'包装', '干净', '卫生', '新鲜', '品质', '质量', '食材', '卫生'}

    taste_count = len(kw_set & taste_words)
    speed_count = len(kw_set & speed_words)
    service_count = len(kw_set & service_words)
    price_count = len(kw_set & price_words)
    quality_count = len(kw_set & quality_words)

    scores = {
        '美食口味体验': taste_count,
        '配送速度效率': speed_count,
        '商家服务态度': service_count,
        '价格性价比': price_count,
        '包装与品质': quality_count,
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return '综合评价'
    return best


# ============ 模型初始化 ============
def init_models():
    """初始化LDA模型和情感分类模型"""
    print("正在初始化模型...")

    try:
        # 加载原始数据
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=['label', 'review'])
        print(f"  数据加载: {len(df)} 条")

        # 预处理文本
        processed_texts = [preprocess_text(text) for text in df['review']]

        # ---- LDA 主题模型 ----
        # 使用CountVectorizer（LDA使用词频矩阵）
        count_vec = CountVectorizer(max_features=2000, min_df=3, max_df=0.9)
        count_matrix = count_vec.fit_transform(processed_texts)
        feature_names = count_vec.get_feature_names()

        lda = LatentDirichletAllocation(
            n_components=5,
            max_iter=30,
            learning_method='online',
            random_state=42
        )
        lda.fit(count_matrix)
        print("  LDA模型训练完成（5个主题）")

        # 提取每个主题的Top15关键词
        topics = []
        all_top_words = {}  # 汇总所有主题关键词用于整体词云
        for idx, topic_dist in enumerate(lda.components_):
            top_indices = topic_dist.argsort()[-15:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_weights = [round(topic_dist[i], 2) for i in top_indices]
            word_freq = {feature_names[i]: float(topic_dist[i]) for i in top_indices}
            all_top_words.update(word_freq)

            # 自动总结主题含义
            meaning = summarize_topic(top_words)

            topics.append({
                'id': idx,
                'name': f'主题 {idx + 1}',
                'label': meaning,
                'keywords': list(zip(top_words, top_weights)),
            })

        # 生成一个整体词云（汇总所有主题关键词）
        overall_wc = generate_wordcloud_base64(all_top_words, color='viridis')
        print("  主题关键词提取完成，整体词云已生成")

        # ---- 情感分类模型 ----
        # 加载已有的TF-IDF特征和标签
        X = joblib.load(os.path.join(FEATURE_DIR, 'X_tfidf.joblib'))
        y = joblib.load(os.path.join(FEATURE_DIR, 'y_labels.joblib'))

        # 训练LinearSVC（使用优化后的参数）
        sentiment_model = LinearSVC(
            random_state=42, max_iter=5000,
            C=0.2, class_weight='balanced'
        )
        sentiment_model.fit(X, y)

        # 加载向量化器（用于新文本预测）
        vectorizer = joblib.load(os.path.join(FEATURE_DIR, 'tfidf_vectorizer.joblib'))
        print("  情感分类模型训练完成")

        return topics, overall_wc, sentiment_model, vectorizer

    except Exception as e:
        print(f"模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return [], None, None, None


# ============ 启动时加载模型 ============
TOPICS, OVERALL_WC, SENTIMENT_MODEL, VECTORIZER = init_models()


# ============ 路由定义 ============

@app.route('/')
def index():
    """首页：主题列表 + 词云图 + 预测入口"""
    return render_template('index.html', topics=TOPICS, overall_wc=OVERALL_WC)


@app.route('/topic/<int:topic_id>')
def topic_detail(topic_id):
    """主题详情页"""
    if 0 <= topic_id < len(TOPICS):
        topic = TOPICS[topic_id]
        return render_template('topic.html', topic=topic, topics=TOPICS)
    return "主题不存在", 404


@app.route('/predict', methods=['POST'])
def predict():
    """情感预测接口"""
    try:
        comment = request.form.get('comment', '').strip()
        if not comment:
            return render_template('index.html', topics=TOPICS, overall_wc=OVERALL_WC,
                                   prediction_text='请输入评论内容',
                                   prediction_type='warning',
                                   input_comment='')

        if SENTIMENT_MODEL is None or VECTORIZER is None:
            return render_template('index.html', topics=TOPICS, overall_wc=OVERALL_WC,
                                   prediction_text='模型未加载，请检查数据文件',
                                   prediction_type='warning',
                                   input_comment=comment)

        # 预处理
        processed = preprocess_text(comment)
        X_new = VECTORIZER.transform([processed])
        pred = SENTIMENT_MODEL.predict(X_new)[0]

        if pred == 1:
            result = '正向评价 (好评)'
            ptype = 'positive'
        else:
            result = '负向评价 (差评)'
            ptype = 'negative'

        return render_template('index.html', topics=TOPICS, overall_wc=OVERALL_WC,
                               prediction_text=result,
                               prediction_type=ptype,
                               input_comment=comment)

    except Exception as e:
        return render_template('index.html', topics=TOPICS, overall_wc=OVERALL_WC,
                               prediction_text=f'预测出错: {e}',
                               prediction_type='warning',
                               input_comment=request.form.get('comment', ''))


# ============ 启动应用 ============
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Flask Web应用启动")
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 50)
    print("学号: 23053080649")
    print("=" * 50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
