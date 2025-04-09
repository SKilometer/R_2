import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
file_path = 'medmcqa.csv'  # 替换为你的CSV文件路径
output_filtered = 'medmcqa_filtered.csv'  # 筛选后的数据输出路径
output_train = 'medmcqa_train.csv'  # 训练集输出路径
output_test = 'medmcqa_test.csv'  # 测试集输出路径

# 读取数据
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"文件 {file_path} 未找到，请检查路径。")
    exit()

# 删除'exp'为空的行
data = data.dropna(subset=['exp'])

# 筛选'exp'列中单词超过10个的行
data = data[data['exp'].str.split().apply(len) > 200]

data = data.sample(n=2000, random_state=42)

# 保存筛选后的数据
data.to_csv(output_filtered, index=False)
print(f"筛选后的数据已保存为: {output_filtered}")

# 使用8:2随机分割
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 保存训练集和测试集为两个CSV文件
train.to_csv(output_train, index=False)
test.to_csv(output_test, index=False)

print(f"训练集已保存为: {output_train}")
print(f"测试集已保存为: {output_test}")