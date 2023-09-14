# coding=utf-8
import glob
import pandas as pd
from scipy.io import arff


def arff_to_csv(arff_path, csv_path):
    """
    :param arff_path: arff文件的路径
    :param csv_path: 生成的csv文件的路径
    :function 将arff文件转换为csv文件,并且对最后一个字段进行处理
    """
    data, meta = arff.loadarff(arff_path)
    csv_data = pd.DataFrame(data)
    # 将Defective标签转换为label标签,对数据也进行转换b'N'->0,b'Y'->1
    if 'label' in csv_data.columns:
        csv_data.loc[csv_data['label'] == b'N'] = 0
        csv_data.loc[csv_data['label'] == b'Y'] = 1
    elif 'Defective' in csv_data.columns:
        csv_data['label'] = -1
        csv_data.loc[csv_data['Defective'] == b'N', 'label'] = 0
        csv_data.loc[csv_data['Defective'] == b'Y', 'label'] = 1
        csv_data.drop(['Defective'], axis=1, inplace=True)
    # print(csv_path)
    csv_data.to_csv(csv_path, index=False)
