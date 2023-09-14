import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

true_props = []
false_props = []
files = glob.glob('./dataset/OriginalData/CSV/*.csv')
file_list = []
for csv_file in files:
    # 读取所有csv文件
    data = pd.read_csv(csv_file)
    print(data.columns)
    # 通过对data中的'label'列进行统计，计算了正类（label为1）和负类（label为0）的数量，并分别存储在true_num和false_num变量中。
    true_num = data.label.value_counts().loc[1]
    false_num = data.label.value_counts().loc[0]
    # print(true_num, false_num, true_num+false_num)
    # 计算了正类和负类的占比（比例），分别存储在true_prop和false_prop变量中
    true_prop = true_num / (true_num + false_num)
    false_prop = false_num / (true_num + false_num)
    # 这三行将计算得到的正类和负类的占比分别添加到true_props和false_props列表中，
    # 并将文件名添加到file_list中。文件名通过从csv_file字符串中截取文件名部分（去除路径和扩展名）获得。
    true_props.append(true_prop)
    false_props.append(false_prop)
    file_list.append(csv_file[27:-4])

# 绘制图形
fig, ax = plt.subplots(dpi=120)
x = np.arange(len(file_list))
width = 0.35
ax.bar(x - width / 2, false_props, width, label='负类')
ax.bar(x + width / 2, true_props, width, label='正类')

ax.legend()
ax.set_xticks(x)
ax.set_xticklabels(file_list)
ax.set_xlabel("数据集")
ax.set_ylabel("占比")
ax.set_title("数据集正负类分布")
fig.tight_layout()
plt.savefig('./visual/distribute.jpg')
