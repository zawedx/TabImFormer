import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import json
import matplotlib.colors
import traceback

print("--- 脚本开始运行 (V4 - 重建颜色数组版) ---")

try:
    # --- 1. 加载和预处理数据 ---
    json_file_path = "/home/two/LMTDE/shap_data_ad.json"
    print(f"正在从以下路径加载数据: {json_file_path}")
    with open(json_file_path, 'r') as f:
        shap_data_dict = json.load(f)

    feature_names = list(shap_data_dict.keys())
    num_features = len(feature_names)
    if not shap_data_dict:
        raise ValueError("SHAP数据字典为空，请检查JSON文件。")
    max_samples = max(len(v) for v in shap_data_dict.values() if v) if any(shap_data_dict.values()) else 0
    print(f"找到 {num_features} 个特征。最长序列长度为 {max_samples}。")

    # --- 2. 将数据转换为SHAP库所需的格式 (带补零) ---
    shap_values = np.zeros((max_samples, num_features))
    feature_values = np.zeros((max_samples, num_features))
    is_padded_mask = np.ones_like(shap_values, dtype=bool)

    for i, name in enumerate(feature_names):
        data_points = shap_data_dict.get(name)
        if not data_points: continue
        f_vals, s_vals = zip(*data_points)
        current_length = len(f_vals)
        feature_values[:current_length, i] = np.array(f_vals)
        shap_values[:current_length, i] = np.array(s_vals)
        is_padded_mask[:current_length, i] = False
        # fill with np.nan for padding
        feature_values[current_length:, i] = np.nan
        # shap_values[current_length:, i] = np.nan
    print("数据已成功转换为补nan的Numpy数组。")

    # --- 3. 特征选择 ---
    top_n = 10
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if num_features < top_n: top_n = num_features
    important_indices = np.argsort(mean_abs_shap)[-top_n:]

    mri_feature_name = next((name for name in feature_names if 'MRI' in name), None)
    print(f"\n[DIAGNOSTIC] 检测到的MRI特征: {mri_feature_name}")

    important_feature_names = [feature_names[i] for i in important_indices]
    if mri_feature_name and mri_feature_name not in important_feature_names:
        important_feature_names.append(mri_feature_name)

    final_indices = [feature_names.index(name) for name in important_feature_names]
    shap_values_subset = shap_values[:, final_indices]
    feature_values_subset = feature_values[:, final_indices].copy()
    is_padded_mask_subset = is_padded_mask[:, final_indices]
    final_feature_names = important_feature_names

    if mri_feature_name:
        mri_index_in_subset = final_feature_names.index(mri_feature_name)
        feature_values_subset[:, mri_index_in_subset] = np.nan

    final_plot_names = []
    plot_name_to_original_name = {}
    for name in final_feature_names:
        plot_name = name.replace('img_MRI_', 'Brain MRI') if 'MRI' in name else name
        final_plot_names.append(plot_name)
        plot_name_to_original_name[plot_name] = name
    print(f"最终选择用于绘图的特征: {final_plot_names}")

    my_cmap = plt.get_cmap('RdBu_r')
    my_cmap.set_bad((0, 0, 0, 0))  # 设置透明色

    # === 在 3. 特征选择之后、4. 绘图之前插入 ===
    # 归一化副本，保留原数组以免影响其他计算
    norm_feature_values_subset = feature_values_subset.copy()

    for j in range(norm_feature_values_subset.shape[1]):    # 遍历每列
        col = norm_feature_values_subset[:, j]
        mask = ~np.isnan(col)                              # 只对非 NaN 做缩放
        if mask.any():                                     # 如果该列不是全 NaN
            vmin = col[mask].min()
            vmax = col[mask].max()
            if vmax > vmin:                                # 避免除零
                norm_feature_values_subset[mask, j] = (
                    (col[mask] - vmin) / (vmax - vmin)
                )
    # ===============================================

    # --- 4. 生成图形并修改颜色 ---
    print("\n正在生成SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    # 下面 summary_plot 使用归一化后的数组
    shap.summary_plot(
        shap_values_subset,
        norm_feature_values_subset,            # ← 改这里
        feature_names=final_plot_names,
        show=False,
        max_display=norm_feature_values_subset.shape[0],
        cmap=my_cmap
    )

    # # --- 5. 添加标题和保存 ---
    plt.xlabel('Shapley value (AD)', fontsize=14)
    plt.tight_layout()
    save_path = "shap_ad.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n--- 脚本运行结束 ---")
    print(f"图形已保存至 '{save_path}'")

except Exception as e:
    print(f"\n--- 脚本发生严重错误 ---")
    traceback.print_exc()
