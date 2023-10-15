import numpy as np
import antropy as ant
from pyts.approximation import PiecewiseAggregateApproximation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import numpy as np
from sklearn.decomposition import PCA
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import EntropyHub as EH
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import time
def read_pickle(work_path):
   data_list = []
   with open(work_path, "rb") as f:
      while True:
         try:
            data = pickle.load(f)
            data_list.append(data)
         except EOFError:
            break
   return data_list
start=time.time()
list1=[]
boxes3 = read_pickle("machine-1-1_test.pkl")
boxes3=np.squeeze(np.array(boxes3))
for j in range(len(boxes3[0,:])):
    qlist = []
    te = boxes3[:, j]  # 保存基本统计量
    maximum = max(te)
    minimum = min(te)
    ystd = np.std(te)
    ymean = np.mean(te)
    threshold1 = ymean - 3 * ystd
    threshold2 = ymean + 3 * ystd
    flag = 0
    hs = 0
    templ=[]
    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            flag = flag + 1
            templ.append(i)
            qlist.append(1)

        else:
            qlist.append(0)

    temp=ant.sample_entropy(qlist)

    list1.append(temp)

end=time.time()
print("程序process_1的运行时间为：{}".format(end-start))
ystd = np.std(list1)
ymean = np.mean(list1)
threshold1 = ymean - 3 * ystd
threshold2 = ymean + 3 * ystd

QL = np.quantile(list1, 0.25, interpolation='lower')  # 下四分位数
QU = np.quantile(list1, 0.75, interpolation='higher')  # 上四分位数
R=maximum-minimum
IQR = QU - QL
threshold1 = QL - 1.5 * (IQR) # 下阈值
threshold2 = QU + 1.5 * (IQR)  # 上阈值

x1=np.linspace(0,37, num=38)
x2=np.linspace(1,38, num=38)
# Draw plot
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)

plt.axhline(y=threshold2, c="g", ls="--", lw=2)
ax.vlines(x=x1, ymin=0, ymax=list1, color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=x1, y=list1, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
# ax.set_title('An example of Sample Entropy from label list of SMD dataset', fontdict={'size':22})
plt.tick_params(labelsize=16)
ax.set_ylabel('Sample Entropy',fontsize=22)
ax.set_xlabel('KPI Dimensions',fontsize=22)
ax.set_xticks(x1)
plt.xticks(rotation=45)
# ax.set_xticklabels(df.manufacturer.str.upper(), rotation=60, fontdict={'horizontalalignment': 'right', 'size':12})
# ax.set_ylim(0, 0.05)

# Annotate
# for row in df.itertuples():
#     ax.text(row.Index, row.cty+.5, s=round(row.cty, 2), horizontalalignment= 'center', verticalalignment='bottom', fontsize=14)

plt.show()