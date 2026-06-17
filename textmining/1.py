#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
外卖评论数据获取脚本
功能：从指定链接获取外卖评论数据集，并进行详细分析
作者：AI助手
日期：2026-06-17
权限声明：本脚本仅用于学习和研究目的，数据来源为公开数据集
"""

import pandas as pd
import requests
from io import StringIO
import sys
import time
import random
import os
import contextlib
import datetime

def fetch_data():
    """获取外卖评论数据集"""
    
    # 数据链接配置
    primary_url = "https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/waimai_10k/waimai_10k.csv"
    backup_url = "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv"
    
    print("=" * 60)
    print("外卖评论数据获取脚本")
    print("=" * 60)
    print(f"主链接: {primary_url}")
    print(f"备用链接: {backup_url}")
    print("=" * 60)
    
    # 权限声明
    print("\n【权限声明】")
    print("1. 本脚本仅用于学习和研究目的")
    print("2. 数据来源为公开数据集，遵循相应开源协议")
    print("3. 请勿将数据用于商业用途或违反相关法律法规")
    print("4. 使用本脚本即表示您同意遵守相关使用条款\n")
    
    # 设置请求头，模拟浏览器访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # 请求延时，遵守爬虫规范
    print("正在添加请求延时...")
    time.sleep(random.uniform(1, 3))

    # 尝试从主链接获取数据
    print("正在尝试从主链接获取数据...")
    try:
        response = requests.get(primary_url, timeout=10, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        
        # 检查响应内容
        if len(response.text) == 0:
            raise ValueError("主链接返回空内容")
        
        print("✓ 主链接访问成功！")
        print(f"响应状态码: {response.status_code}")
        print(f"内容长度: {len(response.text)} 字符")
        
        # 解析CSV数据
        try:
            df = pd.read_csv(StringIO(response.text))
            print("\n✓ 数据解析成功！")
            display_data_details(df)
            return df
        except Exception as e:
            print(f"✗ 数据解析失败: {e}")
            print("尝试使用备用链接...")
            
    except requests.exceptions.Timeout:
        print("✗ 主链接请求超时（10秒）")
        print("尝试使用备用链接...")
    except requests.exceptions.ConnectionError:
        print("✗ 主链接连接失败")
        print("尝试使用备用链接...")
    except requests.exceptions.HTTPError as e:
        print(f"✗ 主链接HTTP错误: {e}")
        print("尝试使用备用链接...")
    except requests.exceptions.RequestException as e:
        print(f"✗ 主链接请求异常: {e}")
        print("尝试使用备用链接...")
    except Exception as e:
        print(f"✗ 主链接未知错误: {e}")
        print("尝试使用备用链接...")
    
    # 请求延时，遵守爬虫规范
    print("\n正在添加请求延时...")
    time.sleep(random.uniform(1, 3))

    # 尝试从备用链接获取数据
    print("正在尝试从备用链接获取数据...")
    try:
        response = requests.get(backup_url, timeout=10, headers=headers)
        response.raise_for_status()
        
        # 检查响应内容
        if len(response.text) == 0:
            raise ValueError("备用链接返回空内容")
        
        print("✓ 备用链接访问成功！")
        print(f"响应状态码: {response.status_code}")
        print(f"内容长度: {len(response.text)} 字符")
        
        # 解析CSV数据
        try:
            df = pd.read_csv(StringIO(response.text))
            print("\n✓ 数据解析成功！")
            display_data_details(df)
            return df
        except Exception as e:
            print(f"✗ 数据解析失败: {e}")
            print("\n无法获取有效数据，请检查网络连接或数据源。")
            return None
            
    except requests.exceptions.Timeout:
        print("✗ 备用链接请求超时（10秒）")
    except requests.exceptions.ConnectionError:
        print("✗ 备用链接连接失败")
    except requests.exceptions.HTTPError as e:
        print(f"✗ 备用链接HTTP错误: {e}")
    except requests.exceptions.RequestException as e:
        print(f"✗ 备用链接请求异常: {e}")
    except Exception as e:
        print(f"✗ 备用链接未知错误: {e}")
    
    print("\n" + "=" * 60)
    print("无法获取数据，请检查：")
    print("1. 网络连接是否正常")
    print("2. 数据源链接是否可用")
    print("3. 是否有防火墙或代理设置阻止访问")
    print("=" * 60)
    return None

def display_data_details(df):
    """显示数据的详细情况"""
    print("\n" + "=" * 60)
    print("数据集详细信息")
    print("=" * 60)
    
    # 基本信息
    print(f"数据形状: {df.shape[0]} 行, {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    
    # 数据类型
    print("\n数据类型:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # 缺失值统计
    print("\n缺失值统计:")
    missing_values = df.isnull().sum()
    for col in df.columns:
        missing_count = missing_values[col]
        if missing_count > 0:
            percentage = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count} ({percentage:.2f}%)")
        else:
            print(f"  {col}: 0 (0.00%)")
    
    # 标签分布（如果存在label列）
    if 'label' in df.columns:
        print("\n标签分布:")
        label_counts = df['label'].value_counts().sort_index()
        total = len(df)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            label_name = "正面" if label == 1 else "负面"
            print(f"  {label} ({label_name}): {count} ({percentage:.2f}%)")
    
    # 评论长度统计（如果存在review列）
    if 'review' in df.columns:
        print("\n评论长度统计:")
        # 计算评论长度
        df['review_length'] = df['review'].astype(str).apply(len)
        avg_length = df['review_length'].mean()
        max_length = df['review_length'].max()
        min_length = df['review_length'].min()
        median_length = df['review_length'].median()
        
        print(f"  平均长度: {avg_length:.2f} 字符")
        print(f"  最大长度: {max_length} 字符")
        print(f"  最小长度: {min_length} 字符")
        print(f"  中位数长度: {median_length:.2f} 字符")
        
        # 删除临时列
        df.drop('review_length', axis=1, inplace=True)
    
    # 数据预览
    print("\n数据预览（前5行）:")
    print("-" * 60)
    for i in range(min(5, len(df))):
        print(f"行 {i+1}:")
        for col in df.columns:
            value = df.iloc[i][col]
            # 如果是评论，截断显示
            if col == 'review' and isinstance(value, str) and len(value) > 100:
                print(f"  {col}: {value[:100]}...")
            else:
                print(f"  {col}: {value}")
        print()
    
    # 数据质量评估
    print("=" * 60)
    print("数据质量评估")
    print("=" * 60)
    
    # 检查重复行
    duplicate_count = df.duplicated().sum()
    print(f"重复行数量: {duplicate_count} ({(duplicate_count/len(df))*100:.2f}%)")
    
    # 检查标签列的唯一值
    if 'label' in df.columns:
        unique_labels = df['label'].unique()
        print(f"标签唯一值: {sorted(unique_labels)}")
        if set(unique_labels) != {0, 1}:
            print("⚠ 警告: 标签列包含非标准值（应为0和1）")
    
    # 检查评论列的空值
    if 'review' in df.columns:
        empty_reviews = df['review'].isnull().sum()
        print(f"空评论数量: {empty_reviews} ({(empty_reviews/len(df))*100:.2f}%)")
        
        # 检查纯数字或特殊字符的评论
        import re
        def is_invalid_review(text):
            if pd.isna(text):
                return True
            text = str(text).strip()
            if len(text) == 0:
                return True
            # 检查是否全是数字或特殊字符
            if re.match(r'^[\d\W]+$', text):
                return True
            return False
        
        invalid_reviews = df['review'].apply(is_invalid_review).sum()
        print(f"无效评论数量: {invalid_reviews} ({(invalid_reviews/len(df))*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("数据获取完成！")
    print("=" * 60)

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
    log_path = os.path.join(script_dir, 'output1.txt')
    
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
                
                try:
                    df = fetch_data()
                    if df is not None:
                        print("\n数据获取成功，可用于后续分析。")
                        # 可以在这里添加保存数据到本地的代码
                        # df.to_csv('waimai_10k.csv', index=False)
                    else:
                        print("\n数据获取失败，请检查网络连接。")
                        sys.exit(1)
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