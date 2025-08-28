import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import json

# # --- 1. 模拟您的数据 ---
# # 假设我们有25个特征和512个样本 (batch_size=512)
# num_features = 25
# num_samples = 512
# feature_names = [f'feature{i+1}' for i in range(num_features)]

# # 创建一个字典来存储模拟数据，结构与您描述的类似
# shap_data_dict = {}
# for name in feature_names:
#     # 为了让重要性有差异，我们给前几个特征的SHAP值乘以一个较大的系数
#     if int(name.replace('feature', '')) <= 5:
#         factor = 5
#     else:
#         # 其他特征的重要性随机
#         factor = np.random.rand() * 2
        
#     shap_values_for_feature = np.random.randn(num_samples) * factor
#     feature_values_for_feature = np.random.rand(num_samples)
    
#     # 存为(feature_value, shap_value)的元组列表
#     shap_data_dict[name] = list(zip(feature_values_for_feature, shap_values_for_feature))

# print("feature1的原始数据结构示例:")
# # 打印前5个示例
# print(str(shap_data_dict['feature1'][:5]).replace("), (", "),\n("))
# print("-" * 30)

# load json shap data from json file
json_file_path = "/home/two/LMTDE/shap_data.json"
with open(json_file_path, 'r') as f:
    shap_data_dict = json.load(f)

# set global variables
num_features = len(shap_data_dict)
num_samples = len(next(iter(shap_data_dict.values())))  # 获取任意一个特征的样本数量
feature_names = list(shap_data_dict.keys())


# --- 2. 将数据转换为SHAP库所需的格式 ---
# SHAP库需要两个Numpy数组：
# - 一个是SHAP值数组 (n_samples, n_features)
# - 另一个是特征值数组 (n_samples, n_features)
shap_values = np.zeros((num_samples, num_features))
feature_values = np.zeros((num_samples, num_features))

# 填充数组
for i, name in enumerate(feature_names):
    f_vals, s_vals = zip(*shap_data_dict[name])
    feature_values[:, i] = f_vals
    shap_values[:, i] = s_vals

print(f"转换后的SHAP值数组维度: {shap_values.shape}")
print(f"转换后的特征值数组维度: {feature_values.shape}")
print("-" * 30)


# --- 3. 使用shap.summary_plot生成图形 ---
print("正在生成SHAP Summary Plot...")

# 创建一个图形实例以获得更好的显示效果
plt.figure(figsize=(10, 8))

# 调用summary_plot函数
shap.summary_plot(
    shap_values, 
    feature_values, 
    feature_names=feature_names,
    max_display=10, # 指定显示前10个最重要的特征
    show=False, # 我们将手动保存和显示图形
)

# 添加标题和调整布局
# plt.title('Shapley value (MCI)', fontsize=16)
plt.xlabel('Shapley value (MCI)', fontsize=14)
plt.tight_layout()

# 保存图形
plt.savefig("shap_mci.png", dpi=300, bbox_inches='tight')
print("\n图形已成功保存为 'shap_mci.png'")

# --- 4. (可选) 手动计算特征重要性以验证排序 ---
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': mean_abs_shap
}).sort_values('importance', ascending=False)

print("\n手动计算的特征重要性排序 (前10):")
print(importance_df.head(10))


