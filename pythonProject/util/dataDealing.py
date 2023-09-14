import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def hot(corr):
    plt.figure(figsize=(35, 30))
    ax = sns.heatmap(corr, center=0)
    plt.savefig("visual/hot.jpg")


def preprocessor(clean_csv_data):
    """
    :param clean_csv_data: 清洗后的csv数据
    :function 数据归一化处理与特征选择
    """
    original_data = pd.read_csv(clean_csv_data)
    # 数据填充在前面清洗数据时已经进行过了，这里不应该重复
    # 查找缺失值的行与列，isnull().values获取行信息，any()获取列信息
    # original_data.isnull().values.any()
    # print(original_data.describe())
    # 获取X值
    original_x = pd.DataFrame(original_data.drop(['label'], axis=1))
    # 对数据进行归一化处理
    scaler = StandardScaler()
    temp_x = scaler.fit_transform(original_x)
    original_x = pd.DataFrame(data=temp_x, columns=original_x.columns)
    # 获取Y值
    original_y = original_data['label']
    original_y = pd.DataFrame(original_y)

    """
         数据可视化
         1.散点图的绘制
         2.热力图的绘制
    """
    print("数据处理中的绘图")
    # 绘制散点图。选取四个属性属性，观察某个属性随着某个属性的变化而变化的情况。
    temp_data = pd.DataFrame(original_data, columns=['HALSTEAD_EFFORT', 'NODE_COUNT', 'LOC_TOTAL', 'label'])
    sns.pairplot(temp_data, kind='reg', diag_kind='kde', )
    plt.savefig("visual/plot.jpg")
    plt.show()
    sns.set()
    # 获取各列之间的相关系数
    corr = original_data.corr()
    # 绘制热力图
    hot(corr)

    # now we resample, and from that we take training and validation sets
    # # 使用SMOTE进行过采样时正样本和负样本要放在一起，生成比例1：1
    # smo = SMOTE(n_jobs=-1)
    # 这里必须是fit_resample()，有些版本是fit_sample()无法运行
    # x_sampling, y_sampling = smo.fit_resample(train_x_data, train_y_data)
    # 进行过采样，解决数据不平衡问题
    sm = SMOTE(random_state=12, sampling_strategy=1.0)
    x, y = sm.fit_resample(original_x, original_y)
    # train:训练集，val:训练过程中的测试集(边训练边看到训练的结果，及时判断学习状态) test:训练模型结束后，用于评价模型结果的测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=12)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.1, random_state=12)

    combined_training_data = x_train.copy()
    combined_training_data['label'] = y_train

    return original_data, x, y, combined_training_data, x_train, x_test, y_train, y_test
