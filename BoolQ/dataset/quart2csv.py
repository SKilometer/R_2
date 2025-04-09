import pandas as pd

# 定义 .parquet 文件的路径
parquet_file_path = 'boolq.parquet'

# 定义输出的 .csv 文件路径
csv_file_path = 'boolq.csv'

# 使用 pandas 读取 .parquet 文件
try:
    df = pd.read_parquet(parquet_file_path)
    df = df.sample(n=2000, random_state=42)
    # 将数据保存为 .csv 文件
    df.to_csv(csv_file_path, index=False)
    print(f"转换完成！CSV 文件已保存到：{csv_file_path}")
except Exception as e:
    print(f"转换失败！错误信息：{e}")
