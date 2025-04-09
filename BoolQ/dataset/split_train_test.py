import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
file_path = 'boolq.csv'  # 替换为你的CSV文件路径
output_train = 'boolq_train.csv'   # 训练集输出路径
output_test = 'boolq_test.csv'    # 测试集输出路径

data = pd.read_csv(file_path)

# 使用8:2随机分割
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 保存训练集和测试集为两个CSV文件
train.to_csv(output_train, index=False)
test.to_csv(output_test, index=False)

print(f"训练集已保存为: {output_train}")
print(f"测试集已保存为: {output_test}")
