import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class ClusterUtil(object):
    def __init__(self, csv_file='tourist_agency_ratio.csv'):
        """
        初始化聚类分析类
        
        Parameters:
        -----------
        csv_file : str
            包含旅行社数据的CSV文件路径
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 读取数据
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        print("数据加载成功！")
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {self.df.columns.tolist()}")
        
        # 选择用于聚类的特征
        self.feature_columns = ['non_weekend_ratio', 'out_province_ratio', 'elderly_ratio']
        self.features = self.df[self.feature_columns].values
        
        # 查看数据基本信息
        print("\n特征数据描述:")
        print(self.df[self.feature_columns].describe())
    
    def get_k_means_clusters(self, n_clusters=6):
        """
        获取KMeans聚类结果
        
        Parameters:
        -----------
        n_clusters : int
            聚类数量，默认6
        """
        print(f"\n开始KMeans聚类分析，聚类数: {n_clusters}")
        
        # 标准化数据
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)
        print("数据标准化完成")
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = kmeans.fit_predict(self.features_scaled)
        print("聚类完成")
        
        # 查看聚类结果
        print("\n聚类结果（旅行社名称和所属簇）:")
        result_df = self.df[['tourist_agency_name', 'cluster']]
        print(result_df.head(10))
        
        # 统计每个簇的旅行社数量
        print("\n各簇旅行社数量统计:")
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"  簇 {cluster_id}: {count} 家旅行社")
        
        # 查看聚类中心（反标准化）
        centers_scaled = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)
        
        print("\n聚类中心（原始尺度）:")
        centers_df = pd.DataFrame(centers, columns=self.feature_columns)
        print(centers_df)
        
        # 保存结果
        self.df.to_csv('tourist_agency_ratio_cluster.csv', index=False, encoding='utf-8-sig')
        print("\n结果已保存到 tourist_agency_ratio_cluster.csv")
        
        # 将聚类中心转化为DataFrame用于绘图
        centers_df = pd.DataFrame(centers, columns=self.feature_columns)
        centers_df['cluster'] = [f'簇{i}' for i in range(centers.shape[0])]
        
        # 转换数据格式为长格式用于绘图
        centers_long = pd.melt(centers_df, id_vars=['cluster'], 
                               var_name='feature', value_name='value')
        
        # 绘制聚类中心柱状图
        colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF69B4', '#FF4500', '#FF0000']
        # 确保颜色数量足够
        if len(colors) < n_clusters:
            colors = colors * (n_clusters // len(colors) + 1)
        colors = colors[:n_clusters]
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='feature', y='value', hue='cluster', data=centers_long, palette=colors)
        plt.title('聚类中心特征对比', fontsize=14, fontweight='bold')
        plt.xlabel('特征', fontsize=12)
        plt.ylabel('值', fontsize=12)
        plt.legend(title='簇', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
        
        # 添加数值标签
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.text(p.get_x() + p.get_width() / 2, height + 0.02, 
                       f'{height:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('cluster_centers.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 输出每个簇的特征分析
        self._analyze_clusters(centers_df)
        
        return self.df, kmeans, scaler
    
    def _analyze_clusters(self, centers_df):
        """
        分析每个簇的特征
        """
        print("\n" + "="*60)
        print("聚类结果分析")
        print("="*60)
        
        feature_names_cn = {
            'non_weekend_ratio': '非周末占比',
            'out_province_ratio': '外省游客占比',
            'elderly_ratio': '老年人占比'
        }
        
        for i, row in centers_df.iterrows():
            print(f"\n【{row['cluster']}】特征分析:")
            print(f"  非周末游客占比: {row['non_weekend_ratio']:.3f}")
            print(f"  外省游客占比: {row['out_province_ratio']:.3f}")
            print(f"  老年人占比: {row['elderly_ratio']:.3f}")
            
            # 特征解读
            features_desc = []
            if row['non_weekend_ratio'] > 0.6:
                features_desc.append("偏向非周末出行")
            elif row['non_weekend_ratio'] < 0.4:
                features_desc.append("偏向周末出行")
            
            if row['out_province_ratio'] > 0.5:
                features_desc.append("外省游客为主")
            else:
                features_desc.append("本地游客为主")
            
            if row['elderly_ratio'] > 0.4:
                features_desc.append("老年游客占比较高")
            
            print(f"  特征解读: {'，'.join(features_desc)}")
        
        # 输出每个簇包含的旅行社
        print("\n" + "="*60)
        print("各簇包含的旅行社")
        print("="*60)
        for cluster_id in sorted(self.df['cluster'].unique()):
            agencies = self.df[self.df['cluster'] == cluster_id]['tourist_agency_name'].tolist()
            print(f"\n簇 {cluster_id} ({len(agencies)}家):")
            for agency in agencies[:5]:  # 只显示前5家
                print(f"  - {agency}")
            if len(agencies) > 5:
                print(f"  ... 还有 {len(agencies)-5} 家")
    
    def find_best_k(self, k_range=range(2, 10)):
        """
        使用肘部法和轮廓系数法寻找最佳K值
        """
        from sklearn.metrics import silhouette_score
        
        print("\n" + "="*60)
        print("寻找最佳K值")
        print("="*60)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.features)
        
        # 肘部法
        sse = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(features_scaled)
            sse.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(features_scaled, kmeans.labels_))
            print(f"K={k}: SSE={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")
        
        # 绘制肘部法曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 肘部法图
        ax1.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('K值')
        ax1.set_ylabel('SSE')
        ax1.set_title('肘部法')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 轮廓系数图
        ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('K值')
        ax2.set_ylabel('轮廓系数')
        ax2.set_title('轮廓系数法')
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        best_k = k_range[np.argmax(silhouette_scores)]
        ax2.axvline(x=best_k, color='red', linestyle='--', label=f'最佳K={best_k}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('best_k_analysis.png', dpi=300)
        plt.show()
        
        print(f"\n建议的最佳K值: {best_k}")
        return best_k


if __name__ == '__main__':
    # 创建聚类分析实例
    cu = ClusterUtil('tourist_agency_ratio.csv')
    
    # 可选：先寻找最佳K值
    # best_k = cu.find_best_k()
    # cu.get_k_means_clusters(n_clusters=best_k)
    
    # 直接使用K=6进行聚类（根据原图片代码）
    cu.get_k_means_clusters(n_clusters=6)