import csv

# --- 配置 ---
# 源 CSV 文件路径
csv_file_path = "/home/two/LMTDE/data/datasets/nacc_new/naccImg_validation_normed.csv.backup"
# 筛选后要保存的目标 CSV 文件路径
target_csv_path = "/home/two/LMTDE/data/datasets/nacc_new/naccImg_validation_normed.csv"
# 需要检查的特征（列名）
feature_to_check = 'img_MRI_mIP'

try:
    # 使用 'with' 语句可以确保文件在使用后被正确关闭
    # newline='' 参数可以防止写入时出现多余的空行
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as source_file, \
         open(target_csv_path, 'w', newline='', encoding='utf-8') as target_file:

        # 创建 CSV 读取器和写入器
        reader = csv.reader(source_file)
        writer = csv.writer(target_file)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        # 查找 'img_MRI_mIP' 列的索引位置
        try:
            feature_index = header.index(feature_to_check)
        except ValueError:
            # 如果在表头中找不到指定的列，则打印错误信息并退出
            print(f"错误：在CSV文件的表头中找不到列 '{feature_to_check}'。")
            exit()

        # 遍历源文件中的每一行
        for row in reader:
            # 检查行长度是否足够，以及目标列的值是否非空
            # `row[feature_index]` 会检查字符串是否不为空
            if len(row) > feature_index and row[feature_index]:
                # 如果该列非空，则将该行写入目标文件
                writer.writerow(row)

    print(f"成功筛选 CSV 文件。")
    print(f"包含非空 '{feature_to_check}' 的行已被保存至：'{target_csv_path}'")

except FileNotFoundError:
    print(f"错误：找不到源文件 '{csv_file_path}'。")
except Exception as e:
    print(f"发生了一个意外错误: {e}")

