# import os.path
# import sys
from util.dataTransfer import arff_to_csv
import glob
import pandas as pd


# 遍历所有arff文件，将其转换成csv文件
for file in glob.glob('../dataset/OriginalData/MDP/*.arff'):
    csv_path = file.replace('arff', 'csv').replace('MDP', 'CSV')
    arff_to_csv(file, csv_path)
# 将所有csv文件合并存储到merge_data中

merge_data = pd.DataFrame()
for csv_file in glob.glob('../dataset/OriginalData/CSV/*.csv'):
    data = pd.read_csv(csv_file)
    merge_data = pd.concat([merge_data, data], ignore_index=True)


# 对数据进行处理
# 丢弃问题数据——去除数据集中的空值NAN
merge_data = merge_data.dropna(thresh=32)
# 查找缺失值的行与列，isnull().values获取行信息，any()获取列信息
# 用0来填充缺失的值
merge_data.fillna(value=0, inplace=True)
# 重置数据索引
merge_data.to_csv("../dataset/CleanedData/clean_data.csv", index=False)
