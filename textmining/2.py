#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本预处理与特征提取脚本
功能：加载CSV数据，jieba分词，停用词过滤，TF-IDF特征提取，保存结果
作者：AI助手
日期：2026-06-17
"""

import os
import sys
import re
import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import contextlib
import datetime


def check_and_install_dependencies():
    """检查并安装依赖库"""
    print("检查依赖库...")
    try:
        import jieba
        import sklearn
        import joblib
        print("✓ 依赖库已安装：jieba, sklearn, joblib")
        return True
    except ImportError as e:
        print(f"✗ 缺少依赖库: {e}")
        print("请在sjwj虚拟环境中执行以下命令安装：")
        print("pip install jieba scikit-learn joblib")
        return False


def load_data(file_path):
    """
    从CSV文件加载数据
    
    参数:
        file_path: CSV文件路径
    
    返回:
        df: 包含label和review列的DataFrame
    """
    print(f"正在加载数据: {file_path}")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要列是否存在
        if 'label' not in df.columns or 'review' not in df.columns:
            raise ValueError("CSV文件必须包含'label'和'review'列")
        
        # 删除缺失值
        initial_count = len(df)
        df = df.dropna(subset=['label', 'review'])
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"已删除 {initial_count - final_count} 行缺失值数据")
        
        print(f"✓ 数据加载成功，共 {len(df)} 条记录")
        print(f"  正面评价: {len(df[df['label'] == 1])} 条")
        print(f"  负面评价: {len(df[df['label'] == 0])} 条")
        
        return df
    
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return None


def get_stopwords():
    """
    获取停用词列表（至少30个常见中文停用词）
    
    返回:
        stopwords: 停用词集合
    """
    # 内置常见中文停用词集合（包含50个高频无意义词）
    stopwords = set([
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
        '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
        '么', '那', '里', '把', '让', '被', '从', '但', '还', '可以',
        '又', '对', '它', '能', '而且', '因为', '所以', '如果', '虽然', '但是',
        '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '哪里', '谁', '何时',
        '多', '少', '大', '小', '上', '下', '左', '右', '前', '后',
        '个', '只', '条', '张', '本', '把', '件', '样', '种', '些',
        '每', '各', '某', '该', '此', '其', '之', '与', '及', '或'
    ])
    
    print(f"✓ 停用词表已加载，共 {len(stopwords)} 个停用词")
    return stopwords


def preprocess_text(text, stopwords):
    """
    文本预处理：分词、过滤停用词、过滤单字词
    
    参数:
        text: 原始文本
        stopwords: 停用词集合
    
    返回:
        processed_text: 处理后的文本（词之间用空格分隔）
    """
    try:
        # 去除特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', str(text))
        
        # jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词和单字词（长度为1）
        filtered_words = [
            word for word in words 
            if word not in stopwords and len(word) > 1 and word.strip()
        ]
        
        return ' '.join(filtered_words)
    
    except Exception as e:
        print(f"文本处理异常: {e}")
        return ''


def extract_tfidf_features(texts, max_features=1000):
    """
    使用TF-IDF提取文本特征
    
    参数:
        texts: 文本列表
        max_features: 最大特征数
    
    返回:
        vectorizer: TF-IDF向量化器
        X: 特征矩阵
    """
    print(f"正在提取TF-IDF特征，max_features={max_features}...")
    
    try:
        # 初始化TF-IDF向量化器
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,          # 最小文档频率
            max_df=0.95,       # 最大文档频率
            sublinear_tf=True  # 使用亚线性TF缩放
        )
        
        # 拟合转换文本
        X = vectorizer.fit_transform(texts)
        
        print(f"✓ TF-IDF特征提取完成")
        print(f"  特征矩阵形状: {X.shape}")
        
        return vectorizer, X
    
    except Exception as e:
        print(f"✗ TF-IDF特征提取失败: {e}")
        return None, None


def get_top_features(vectorizer, X, top_n=20):
    """
    获取Top-N重要特征词
    
    参数:
        vectorizer: TF-IDF向量化器
        X: 特征矩阵
        top_n: 返回前N个特征
    """
    try:
        # 获取特征名称（兼容新旧版本sklearn）
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            feature_names = vectorizer.get_feature_names()
        
        # 计算每个特征的平均TF-IDF值
        avg_tfidf = np.array(X.mean(axis=0)).flatten()
        
        # 获取Top-N特征索引
        top_indices = avg_tfidf.argsort()[-top_n:][::-1]
        
        print(f"\n{'='*60}")
        print(f"Top {top_n} 特征词及其平均TF-IDF值")
        print(f"{'='*60}")
        
        for i, idx in enumerate(top_indices, 1):
            print(f"{i:2d}. {feature_names[idx]:15s}  TF-IDF: {avg_tfidf[idx]:.4f}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"获取Top特征词失败: {e}")


def save_results(vectorizer, X, y, output_dir):
    """
    保存处理结果到本地文件
    
    参数:
        vectorizer: TF-IDF向量化器
        X: 特征矩阵
        y: 标签数组
        output_dir: 输出目录
    """
    print(f"\n正在保存结果到: {output_dir}")
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存vectorizer
        vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
        joblib.dump(vectorizer, vectorizer_path)
        print(f"  ✓ vectorizer已保存: {vectorizer_path}")
        
        # 保存特征矩阵
        X_path = os.path.join(output_dir, 'X_tfidf.joblib')
        joblib.dump(X, X_path)
        print(f"  ✓ 特征矩阵X已保存: {X_path}")
        
        # 保存标签
        y_path = os.path.join(output_dir, 'y_labels.joblib')
        joblib.dump(y, y_path)
        print(f"  ✓ 标签y已保存: {y_path}")
        
        print("✓ 所有结果保存完成")
        
    except Exception as e:
        print(f"✗ 保存结果失败: {e}")


class Tee:
    """标准输出分流器，同时输出到控制台和文件"""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

def main():
    """主函数"""
    # 获取当前脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'output2.txt')
    
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # 初始化Tee对象，同时输出到控制台和文件
            tee = Tee(log_file, sys.stdout)
            
            # 重定向标准输出
            with contextlib.redirect_stdout(tee):
                print(f"{'='*60}")
                print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"日志保存路径: {log_path}")
                print(f"{'='*60}")
                
                print("=" * 60)
                print("文本预处理与特征提取脚本")
                print("=" * 60)
                
                try:
                    # 检查依赖
                    if not check_and_install_dependencies():
                        sys.exit(1)
                    
                    # 设置路径
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    data_file = os.path.join(base_dir, 'waimai_10k.csv')
                    output_dir = os.path.join(base_dir, 'tfidf_features')
                    
                    # 检查数据文件是否存在
                    if not os.path.exists(data_file):
                        print(f"✗ 数据文件不存在: {data_file}")
                        print("请先运行1.py获取数据")
                        sys.exit(1)
                    
                    # 加载数据
                    df = load_data(data_file)
                    if df is None:
                        sys.exit(1)
                    
                    # 获取停用词表
                    stopwords = get_stopwords()
                    
                    # 文本预处理（分词、过滤停用词、过滤单字词）
                    print("\n正在进行文本预处理...")
                    print("  - jieba分词")
                    print("  - 过滤停用词")
                    print("  - 过滤单字词（长度为1）")
                    
                    processed_texts = []
                    for i, text in enumerate(df['review']):
                        if (i + 1) % 1000 == 0:
                            print(f"  已处理 {i+1}/{len(df)} 条...")
                        processed_texts.append(preprocess_text(text, stopwords))
                    
                    print(f"✓ 文本预处理完成，共处理 {len(processed_texts)} 条文本")
                    
                    # 提取TF-IDF特征
                    vectorizer, X = extract_tfidf_features(processed_texts, max_features=1000)
                    
                    if vectorizer is None or X is None:
                        sys.exit(1)
                    
                    # 获取标签
                    y = df['label'].values
                    
                    # 输出特征矩阵形状
                    print(f"\n特征矩阵形状: {X.shape}")
                    print(f"  样本数: {X.shape[0]}")
                    print(f"  特征数: {X.shape[1]}")
                    
                    # 获取Top20特征词
                    get_top_features(vectorizer, X, top_n=20)
                    
                    # 保存结果
                    save_results(vectorizer, X, y, output_dir)
                    
                    print("\n" + "=" * 60)
                    print("文本预处理完成！")
                    print("=" * 60)
                    
                except KeyboardInterrupt:
                    print("\n\n用户中断操作。")
                    sys.exit(0)
                except Exception as e:
                    print(f"\n程序执行过程中发生错误: {e}")
                    sys.exit(1)
                finally:
                    # 输出学号
                    print("\n" + "=" * 60)
                    print("学号: 23053080649")
                    print("=" * 60)
                    
        print(f"日志已保存至: {log_path}")
                    
    except Exception as e:
        print(f"日志文件写入失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
