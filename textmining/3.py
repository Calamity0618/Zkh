#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感二分类模型训练与评估脚本
功能：加载TF-IDF特征，训练MultinomialNB和LinearSVC，评估对比
包含：5折交叉验证、错误样本分析、模型优化
作者：AI助手
日期：2026-06-17
"""

import os
import sys
import datetime
import contextlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# ============ matplotlib 中文显示配置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


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


def load_features(feature_dir):
    """
    从Joblib文件加载TF-IDF特征矩阵和标签

    参数:
        feature_dir: 特征文件目录

    返回:
        X: TF-IDF特征矩阵
        y: 标签数组
    """
    print("正在加载特征数据...")

    try:
        X_path = os.path.join(feature_dir, 'X_tfidf.joblib')
        y_path = os.path.join(feature_dir, 'y_labels.joblib')

        # 检查文件是否存在
        if not os.path.exists(X_path):
            raise FileNotFoundError(f"特征矩阵文件不存在: {X_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"标签文件不存在: {y_path}")

        X = joblib.load(X_path)
        y = joblib.load(y_path)

        print(f"✓ 特征数据加载成功")
        print(f"  特征矩阵形状: {X.shape}")
        print(f"  标签数量: {len(y)}")
        print(f"  正面样本: {np.sum(y == 1)}")
        print(f"  负面样本: {np.sum(y == 0)}")

        return X, y

    except Exception as e:
        print(f"✗ 特征数据加载失败: {e}")
        return None, None


def load_raw_data(data_path):
    """
    加载原始CSV数据（用于错误样本分析时显示原文）

    参数:
        data_path: CSV文件路径

    返回:
        df: 原始数据DataFrame
    """
    try:
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print(f"✓ 原始数据加载成功: {len(df)} 条")
            return df
        else:
            print(f"⚠ 原始数据文件不存在: {data_path}")
            return None
    except Exception as e:
        print(f"⚠ 原始数据加载失败: {e}")
        return None


def split_data(X, y, test_size=0.2, random_state=42):
    """
    划分训练集和测试集

    参数:
        X: 特征矩阵
        y: 标签
        test_size: 测试集比例
        random_state: 随机种子

    返回:
        X_train, X_test, y_train, y_test, indices_train, indices_test
    """
    print(f"\n正在划分数据集（训练:测试 = {1-test_size}:{test_size}）...")

    try:
        # 生成索引数组，用于追踪样本位置
        indices = np.arange(len(y))
        
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # 分层抽样，保持类别比例
        )

        print(f"✓ 数据划分完成")
        print(f"  训练集: {X_train.shape[0]} 条（正面: {np.sum(y_train==1)}, 负面: {np.sum(y_train==0)}）")
        print(f"  测试集: {X_test.shape[0]} 条（正面: {np.sum(y_test==1)}, 负面: {np.sum(y_test==0)}）")

        return X_train, X_test, y_train, y_test, idx_train, idx_test

    except Exception as e:
        print(f"✗ 数据划分失败: {e}")
        return None, None, None, None, None, None


def train_mnb(X_train, y_train):
    """
    训练MultinomialNB模型

    参数:
        X_train: 训练特征
        y_train: 训练标签

    返回:
        model: 训练好的模型
    """
    print("\n" + "=" * 60)
    print("训练 MultinomialNB 模型")
    print("=" * 60)

    try:
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        print("✓ MultinomialNB 训练完成")
        return model

    except Exception as e:
        print(f"✗ MultinomialNB 训练失败: {e}")
        return None


def train_svc(X_train, y_train, max_iter=1000, C=1.0):
    """
    训练LinearSVC模型，不收敛时自动优化

    参数:
        X_train: 训练特征
        y_train: 训练标签
        max_iter: 最大迭代次数
        C: 正则化参数

    返回:
        model: 训练好的模型
    """
    print("\n" + "=" * 60)
    print("训练 LinearSVC 模型")
    print("=" * 60)

    try:
        # 第一次尝试，默认参数
        model = LinearSVC(random_state=42, max_iter=max_iter, C=C)
        model.fit(X_train, y_train)
        print(f"✓ LinearSVC 训练完成（max_iter={max_iter}, C={C}）")
        return model

    except Exception as e:
        error_msg = str(e)
        print(f"⚠ LinearSVC 默认参数未收敛: {error_msg}")
        print("正在自动优化参数...")

        # 自动优化：逐步增加 max_iter 和调整参数
        configs = [
            {'max_iter': 5000, 'C': 0.5},
            {'max_iter': 10000, 'C': 0.1},
            {'max_iter': 20000, 'C': 0.05},
        ]

        for i, config in enumerate(configs):
            try:
                print(f"  尝试配置 {i+1}: max_iter={config['max_iter']}, C={config['C']}")
                model = LinearSVC(
                    random_state=42,
                    max_iter=config['max_iter'],
                    C=config['C']
                )
                model.fit(X_train, y_train)
                print(f"✓ LinearSVC 优化训练完成（max_iter={config['max_iter']}, C={config['C']}）")
                return model
            except Exception as e2:
                print(f"    仍未收敛: {e2}")

        print("✗ LinearSVC 所有优化配置均未收敛")
        return None


def evaluate_model(model, X_test, y_test, model_name):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        model_name: 模型名称

    返回:
        metrics: 评估指标字典
        y_pred: 预测结果
    """
    print(f"\n正在评估 {model_name}...")

    try:
        y_pred = model.predict(X_test)

        metrics = {
            '模型名称': model_name,
            '准确率': accuracy_score(y_test, y_pred),
            '精确率': precision_score(y_test, y_pred, average='weighted'),
            '召回率': recall_score(y_test, y_pred, average='weighted'),
            'F1值': f1_score(y_test, y_pred, average='weighted')
        }

        # 打印分类报告
        print(f"\n{model_name} 分类报告:")
        print(classification_report(y_test, y_pred, target_names=['负面(0)', '正面(1)']))

        return metrics, y_pred

    except Exception as e:
        print(f"✗ {model_name} 评估失败: {e}")
        return None, None


def cross_validate_models(X, y, n_folds=5):
    """
    5折交叉验证

    参数:
        X: 特征矩阵
        y: 标签
        n_folds: 折数
    """
    print("\n" + "=" * 60)
    print(f"{n_folds} 折交叉验证")
    print("=" * 60)

    try:
        # 分层K折交叉验证
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        models = {
            'MultinomialNB': MultinomialNB(alpha=1.0),
            'LinearSVC': LinearSVC(random_state=42, max_iter=1000)
        }

        cv_results = {}

        for name, model in models.items():
            print(f"\n{name} 交叉验证结果:")
            print("-" * 40)

            # 计算每折的F1值
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # 训练模型
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)

                # 预测并计算F1
                y_pred_fold = model_copy.predict(X_val_fold)
                f1 = f1_score(y_val_fold, y_pred_fold, average='weighted')
                fold_scores.append(f1)

                print(f"  第 {fold} 折 F1值: {f1:.4f}")

            # 计算平均值和标准差
            mean_f1 = np.mean(fold_scores)
            std_f1 = np.std(fold_scores)

            print("-" * 40)
            print(f"  平均 F1值: {mean_f1:.4f} ± {std_f1:.4f}")

            cv_results[name] = {
                'fold_scores': fold_scores,
                'mean_f1': mean_f1,
                'std_f1': std_f1
            }

        # 对比表格
        print("\n" + "=" * 50)
        print("交叉验证结果对比")
        print("=" * 50)
        print(f"{'模型名称':<20} {'平均F1值':<15} {'标准差':<15}")
        print("-" * 50)
        for name, result in cv_results.items():
            print(f"{name:<20} {result['mean_f1']:<15.4f} {result['std_f1']:<15.4f}")
        print("=" * 50)

        return cv_results

    except Exception as e:
        print(f"✗ 交叉验证失败: {e}")
        return None


def analyze_wrong_samples(model, X_test, y_test, idx_test, raw_df, n_samples=10):
    """
    分析预测错误的样本

    参数:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        idx_test: 测试集在原始数据中的索引
        raw_df: 原始数据DataFrame
        n_samples: 显示的错误样本数量

    返回:
        error_analysis: 错误分析结果
    """
    print("\n" + "=" * 60)
    print("错误样本分析")
    print("=" * 60)

    try:
        # 预测
        y_pred = model.predict(X_test)

        # 找出预测错误的样本
        wrong_mask = y_pred != y_test
        wrong_indices = np.where(wrong_mask)[0]

        total_wrong = len(wrong_indices)
        total_samples = len(y_test)
        error_rate = total_wrong / total_samples * 100

        print(f"\n总样本数: {total_samples}")
        print(f"预测错误数: {total_wrong}")
        print(f"错误率: {error_rate:.2f}%")

        # 错误类型统计
        # 实际为正面(1)但预测为负面(0) - 假阴性
        false_negative = np.sum((y_pred == 0) & (y_test == 1))
        # 实际为负面(0)但预测为正面(1) - 假阳性
        false_positive = np.sum((y_pred == 1) & (y_test == 0))

        print(f"\n错误类型分布:")
        print(f"  假阴性（实际正面→预测负面）: {false_negative} 条")
        print(f"  假阳性（实际负面→预测正面）: {false_positive} 条")

        # 错误样本的长度分析
        print(f"\n错误样本特征分析:")

        # 如果有原始数据，分析文本长度
        if raw_df is not None and 'review' in raw_df.columns:
            # 获取错误样本的原始索引
            wrong_original_indices = idx_test[wrong_indices]

            # 获取正确样本的索引
            correct_indices = np.where(~wrong_mask)[0]
            correct_original_indices = idx_test[correct_indices]

            # 计算文本长度
            wrong_texts = raw_df.iloc[wrong_original_indices]['review'].astype(str)
            correct_texts = raw_df.iloc[correct_original_indices]['review'].astype(str)

            wrong_lengths = wrong_texts.apply(len)
            correct_lengths = correct_texts.apply(len)

            print(f"  错误样本平均文本长度: {wrong_lengths.mean():.1f} 字符")
            print(f"  正确样本平均文本长度: {correct_lengths.mean():.1f} 字符")

            # 文本长度分布
            print(f"\n  错误样本文本长度分布:")
            length_ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 500)]
            for low, high in length_ranges:
                count = np.sum((wrong_lengths >= low) & (wrong_lengths < high))
                pct = count / total_wrong * 100 if total_wrong > 0 else 0
                print(f"    {low}-{high}字符: {count} 条 ({pct:.1f}%)")

            # 显示错误样本示例
            print(f"\n{'='*60}")
            print(f"错误样本示例（前 {min(n_samples, total_wrong)} 条）:")
            print(f"{'='*60}")

            # 随机选择错误样本展示
            np.random.seed(42)
            show_indices = np.random.choice(wrong_indices, min(n_samples, total_wrong), replace=False)

            for i, idx in enumerate(show_indices, 1):
                original_idx = idx_test[idx]
                text = raw_df.iloc[original_idx]['review']
                true_label = y_test[idx]
                pred_label = y_pred[idx]

                # 截断过长的文本
                if len(str(text)) > 80:
                    text_display = str(text)[:80] + "..."
                else:
                    text_display = str(text)

                print(f"\n样本 {i}:")
                print(f"  原始索引: {original_idx}")
                print(f"  文本: {text_display}")
                print(f"  真实标签: {true_label} ({'正面' if true_label == 1 else '负面'})")
                print(f"  预测标签: {pred_label} ({'正面' if pred_label == 1 else '负面'})")
                print(f"  错误类型: {'假阴性' if true_label == 1 else '假阳性'}")

            # 分析错误样本的共性
            print(f"\n{'='*60}")
            print("错误样本共性分析:")
            print(f"{'='*60}")

            # 分析假阴性（正面被误判为负面）
            fn_mask = (y_pred == 0) & (y_test == 1)
            if np.sum(fn_mask) > 0:
                fn_original_indices = idx_test[np.where(fn_mask)[0]]
                fn_texts = raw_df.iloc[fn_original_indices]['review'].astype(str)

                print(f"\n【假阴性样本特征】（正面被误判为负面）:")
                print(f"  平均长度: {fn_texts.apply(len).mean():.1f} 字符")
                print(f"  典型特征:")
                # 检查是否包含感叹号
                exclamation_count = fn_texts.str.contains('！|!').sum()
                print(f"    包含感叹号: {exclamation_count} 条 ({exclamation_count/len(fn_texts)*100:.1f}%)")
                # 检查是否较短
                short_count = (fn_texts.apply(len) < 15).sum()
                print(f"    短文本(<15字): {short_count} 条 ({short_count/len(fn_texts)*100:.1f}%)")

            # 分析假阳性（负面被误判为正面）
            fp_mask = (y_pred == 1) & (y_test == 0)
            if np.sum(fp_mask) > 0:
                fp_original_indices = idx_test[np.where(fp_mask)[0]]
                fp_texts = raw_df.iloc[fp_original_indices]['review'].astype(str)

                print(f"\n【假阳性样本特征】（负面被误判为正面）:")
                print(f"  平均长度: {fp_texts.apply(len).mean():.1f} 字符")
                print(f"  典型特征:")
                # 检查是否包含正面词汇
                positive_words = ['好吃', '不错', '好', '快', '喜欢']
                for word in positive_words:
                    count = fp_texts.str.contains(word).sum()
                    if count > 0:
                        print(f"    包含'{word}': {count} 条")

        else:
            print("  ⚠ 无法加载原始数据，跳过文本分析")

        return {
            'total_wrong': total_wrong,
            'error_rate': error_rate,
            'false_negative': false_negative,
            'false_positive': false_positive,
            'wrong_indices': wrong_indices
        }

    except Exception as e:
        print(f"✗ 错误样本分析失败: {e}")
        return None


def optimized_model_training(X_train, y_train, error_analysis):
    """
    针对错误样本特征进行模型优化

    参数:
        X_train: 训练特征
        y_train: 训练标签
        error_analysis: 错误分析结果

    返回:
        optimized_model: 优化后的模型
    """
    print("\n" + "=" * 60)
    print("模型优化（针对错误样本特征）")
    print("=" * 60)

    try:
        # 根据错误分析结果，调整模型参数
        # 如果假阴性较多（正面被误判为负面），降低正则化强度
        # 如果假阳性较多（负面被误判为正面），增加正则化强度

        fn_ratio = error_analysis['false_negative'] / error_analysis['total_wrong']
        fp_ratio = error_analysis['false_positive'] / error_analysis['total_wrong']

        print(f"\n错误样本分析:")
        print(f"  假阴性比例: {fn_ratio:.2%}")
        print(f"  假阳性比例: {fp_ratio:.2%}")

        # 优化策略
        if fn_ratio > 0.6:
            # 假阴性较多，模型可能过于保守
            print("\n优化策略: 降低正则化强度，增加模型对正面样本的敏感度")
            C_values = [0.5, 0.3, 0.2]
        elif fp_ratio > 0.6:
            # 假阳性较多，模型可能过于激进
            print("\n优化策略: 增加正则化强度，减少误判为正面的情况")
            C_values = [2.0, 3.0, 5.0]
        else:
            # 错误分布较均匀，尝试多种参数
            print("\n优化策略: 尝试多种参数组合寻找最优配置")
            C_values = [0.1, 0.5, 1.0, 2.0]

        best_model = None
        best_f1 = 0
        best_config = {}

        print("\n参数搜索:")
        print("-" * 40)

        for C in C_values:
            for max_iter in [1000, 5000]:
                try:
                    model = LinearSVC(
                        random_state=42,
                        max_iter=max_iter,
                        C=C,
                        class_weight='balanced'  # 处理类别不平衡
                    )
                    model.fit(X_train, y_train)

                    # 使用训练集上的交叉验证评估
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=3, scoring='f1_weighted'
                    )
                    mean_f1 = cv_scores.mean()

                    print(f"  C={C}, max_iter={max_iter}: F1={mean_f1:.4f}")

                    if mean_f1 > best_f1:
                        best_f1 = mean_f1
                        best_model = model
                        best_config = {'C': C, 'max_iter': max_iter}

                except Exception as e:
                    print(f"  C={C}, max_iter={max_iter}: 训练失败 - {e}")

        if best_model is not None:
            print(f"\n✓ 优化完成")
            print(f"  最优参数: C={best_config['C']}, max_iter={best_config['max_iter']}")
            print(f"  最优F1值: {best_f1:.4f}")
            return best_model
        else:
            print("\n✗ 优化失败，使用默认模型")
            return None

    except Exception as e:
        print(f"✗ 模型优化失败: {e}")
        return None


def print_comparison_table(metrics_list):
    """
    打印模型对比表格

    参数:
        metrics_list: 指标字典列表
    """
    print("\n" + "=" * 70)
    print("模型性能对比表")
    print("=" * 70)

    # 表头
    header = f"{'模型名称':<20} {'准确率':<12} {'精确率':<12} {'召回率':<12} {'F1值':<12}"
    print(header)
    print("-" * 70)

    # 数据行
    for metrics in metrics_list:
        row = (f"{metrics['模型名称']:<20} "
               f"{metrics['准确率']:<12.4f} "
               f"{metrics['精确率']:<12.4f} "
               f"{metrics['召回率']:<12.4f} "
               f"{metrics['F1值']:<12.4f}")
        print(row)

    print("=" * 70)

    # 找出最优模型
    best_idx = max(range(len(metrics_list)), key=lambda i: metrics_list[i]['F1值'])
    print(f"\n★ 最优模型（按F1值）: {metrics_list[best_idx]['模型名称']}")


def plot_confusion_matrices(y_test, preds_dict, save_path):
    """
    绘制混淆矩阵热力图（subplot并排）

    参数:
        y_test: 真实标签
        preds_dict: {'模型名': y_pred} 字典
        save_path: 保存路径
    """
    print("\n正在绘制混淆矩阵热力图...")

    try:
        n_models = len(preds_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

        if n_models == 1:
            axes = [axes]

        labels = ['负面(0)', '正面(1)']

        for ax, (name, y_pred) in zip(axes, preds_dict.items()):
            cm = confusion_matrix(y_test, y_pred)

            # 绘制热力图
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # 设置刻度
            ax.set(xticks=[0, 1], yticks=[0, 1],
                   xticklabels=labels, yticklabels=labels,
                   title=f'{name} 混淆矩阵',
                   ylabel='真实标签', xlabel='预测标签')

            # 在格子中显示数值
            thresh = cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=14)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 混淆矩阵热力图已保存: {save_path}")

    except Exception as e:
        print(f"✗ 混淆矩阵绘制失败: {e}")


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, 'output3.txt')

    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            tee = Tee(log_file, sys.stdout)
            with contextlib.redirect_stdout(tee):
                print(f"{'='*60}")
                print(f"运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"日志保存路径: {log_path}")
                print(f"{'='*60}")
                print("=" * 60)
                print("情感二分类模型训练与评估脚本")
                print("=" * 60)

                try:
                    # 1. 加载特征数据
                    feature_dir = os.path.join(script_dir, 'tfidf_features')
                    X, y = load_features(feature_dir)
                    if X is None or y is None:
                        sys.exit(1)

                    # 1.5 加载原始数据（用于错误样本分析）
                    raw_data_path = os.path.join(script_dir, 'waimai_10k.csv')
                    raw_df = load_raw_data(raw_data_path)

                    # 2. 划分训练集和测试集（8:2，stratify=y）
                    X_train, X_test, y_train, y_test, idx_train, idx_test = split_data(
                        X, y, test_size=0.2, random_state=42
                    )
                    if X_train is None:
                        sys.exit(1)

                    # 3. 训练基础模型
                    mnb_model = train_mnb(X_train, y_train)
                    svc_model = train_svc(X_train, y_train)

                    # 4. 评估基础模型
                    metrics_list = []
                    preds_dict = {}

                    if mnb_model is not None:
                        mnb_metrics, mnb_pred = evaluate_model(
                            mnb_model, X_test, y_test, 'MultinomialNB'
                        )
                        if mnb_metrics:
                            metrics_list.append(mnb_metrics)
                            preds_dict['MultinomialNB'] = mnb_pred

                    if svc_model is not None:
                        svc_metrics, svc_pred = evaluate_model(
                            svc_model, X_test, y_test, 'LinearSVC'
                        )
                        if svc_metrics:
                            metrics_list.append(svc_metrics)
                            preds_dict['LinearSVC'] = svc_pred

                    # 5. 打印对比表格
                    if metrics_list:
                        print_comparison_table(metrics_list)

                    # 6. 绘制混淆矩阵热力图
                    if preds_dict:
                        cm_path = os.path.join(script_dir, 'confusion_matrices.png')
                        plot_confusion_matrices(y_test, preds_dict, cm_path)

                    # ========== 新增功能 ==========

                    # 7. 5折交叉验证
                    cross_validate_models(X, y, n_folds=5)

                    # 8. 错误样本分析（使用LinearSVC）
                    best_base_model = svc_model if svc_model is not None else mnb_model
                    error_analysis = None
                    if best_base_model is not None:
                        error_analysis = analyze_wrong_samples(
                            best_base_model, X_test, y_test,
                            idx_test, raw_df, n_samples=10
                        )

                    # 9. 针对错误样本共性进行模型优化
                    if error_analysis is not None:
                        opt_model = optimized_model_training(
                            X_train, y_train, error_analysis
                        )

                        if opt_model is not None:
                            # 评估优化后的模型
                            opt_metrics, opt_pred = evaluate_model(
                                opt_model, X_test, y_test, '优化LinearSVC'
                            )

                            if opt_metrics:
                                metrics_list.append(opt_metrics)
                                preds_dict['优化LinearSVC'] = opt_pred

                                # 打印最终对比表
                                print("\n" + "=" * 70)
                                print("最终模型性能对比表（含优化模型）")
                                print("=" * 70)
                                print_comparison_table(metrics_list)

                                # 更新混淆矩阵图（含优化模型）
                                cm_path_opt = os.path.join(
                                    script_dir, 'confusion_matrices_optimized.png'
                                )
                                plot_confusion_matrices(y_test, preds_dict, cm_path_opt)

                    print("\n" + "=" * 60)
                    print("模型训练与评估完成！")
                    print("=" * 60)

                except KeyboardInterrupt:
                    print("\n\n用户中断操作。")
                    sys.exit(0)
                except Exception as e:
                    print(f"\n程序执行过程中发生错误: {e}")
                    sys.exit(1)
                finally:
                    print("\n" + "=" * 60)
                    print("学号: 23053080649")
                    print("=" * 60)

        print(f"日志已保存至: {log_path}")

    except Exception as e:
        print(f"日志文件写入失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()