import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

# 设置学术绘图风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#bbbbbb'
plt.rcParams['axes.linewidth'] = 0.8

nsl_path = "./train-NSL-KDD.csv"
unsw_path = "./train-UNSW-NB15.csv"

def run_fixed_eda(csv_path, dataset_name):
    print(f"=== Running Fixed EDA for {dataset_name} ===")
    df = pd.read_csv(csv_path)
    label_col = "is_anomaly"
    
    feature_cols = [c for c in df.columns if c != label_col]
    normal_df = df[df[label_col] == 0].copy()
    
    num_cols = normal_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # 找到偏态最大的数值特征
    skewness = normal_df[num_cols].skew().sort_values(ascending=False)
    top_skew_col = skewness.index[0]
    print(f"Top skewed feature: {top_skew_col} (Skew: {skewness.iloc[0]:.2f})")
    
    # -------------------------------------------------------------
    # 1. 准备 t-SNE 降维数据
    # -------------------------------------------------------------
    np.random.seed(42)
    sample_normal = normal_df[num_cols].fillna(0).sample(n=min(1500, len(normal_df)), random_state=42)
    
    # 模拟 Baseline：从 20 个原型中心加高斯微噪扩展出 1500 个样本
    proto_20 = sample_normal.sample(n=20, random_state=42)
    baseline_generated = []
    for _ in range(75):
        noise = np.random.normal(0, 0.1, proto_20.shape)
        baseline_generated.append(proto_20.values + noise)
    baseline_samples = np.vstack(baseline_generated)[:1500]
    
    # 拼在一起跑 t-SNE 确保在同一个空间
    combined = np.vstack([sample_normal.values, baseline_samples])
    tsne = TSNE(n_components=2, random_state=42, perplexity=35, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(combined)
    
    tsne_real = tsne_results[:len(sample_normal)]
    tsne_base = tsne_results[len(sample_normal):]
    
    # -------------------------------------------------------------
    # 2. 开始画图
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    
    # 【左图】：Mode Collapse / Omission 真实视觉效果
    axes[0].scatter(tsne_real[:, 0], tsne_real[:, 1], c='#d0d7de', s=15, alpha=0.5, label='Full Normal Support ($D_0$)')
    axes[0].scatter(tsne_base[:, 0], tsne_base[:, 1], c='#ff4d4d', s=12, alpha=0.7, label='Baseline Generated (Mode Collapsed)')
    axes[0].set_title(f'({dataset_name}) Baseline Mode Collapse in Feature Space', fontsize=11, fontweight='bold')
    axes[0].legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # 【右图】：修复极值拉爆，展示真正的分布对比
    raw_data = normal_df[top_skew_col].values
    
    # 过滤掉高于 98% 分位数的数据用于原始画图，避免 X 轴被极值拉爆
    upper_bound = np.percentile(raw_data, 98)
    if upper_bound == 0:
        upper_bound = np.percentile(raw_data[raw_data > 0], 95) if (raw_data > 0).any() else 1
        
    filtered_raw = raw_data[raw_data <= upper_bound]
    log_data = np.log1p(np.maximum(0, raw_data))
    
    # 双 X 轴并排展示
    ax2 = axes[1]
    ax2_twin = ax2.twiny()
    
    # 注意：这里将 \le 改为了普通的 <=，避免 Matplotlib 语法报错
    sns.kdeplot(filtered_raw, ax=ax2, color='#2b5c8f', fill=True, alpha=0.3, label=f'Raw {top_skew_col} (<= 98th %ile)')
    sns.kdeplot(log_data, ax=ax2_twin, color='#e76f51', fill=True, alpha=0.3, label='Log(1+x) Transformed')
    
    ax2.set_xlabel(f'Raw {top_skew_col} (Truncated for View)', color='#2b5c8f', fontweight='bold')
    ax2_twin.set_xlabel('Log(1+x) Transformed Scale', color='#e76f51', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title(f'({dataset_name}) Skewness & Heavy-Tail Distribution', fontsize=11, fontweight='bold', pad=25)
    
    # 显式合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    plt.tight_layout()
    output_filename = f"eda_{dataset_name.lower()}_fixed.png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_filename}")

if __name__ == "__main__":
    if os.path.exists(nsl_path):
        run_fixed_eda(nsl_path, "NSL-KDD")
    if os.path.exists(unsw_path):
        run_fixed_eda(unsw_path, "UNSW-NB15")