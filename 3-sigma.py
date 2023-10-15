import numpy as np
import antropy as ant
from pyts.approximation import PiecewiseAggregateApproximation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
import pandas as pd
import numpy as np

import pickle
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
    x = np.linspace(1, 28479, num=28479)
    plt.tick_params(labelsize=12)
    plt.plot(x, te)
    plt.axhline(y=threshold2, c="r", ls="--", lw=2)
    outlier = []  # 将异常值保存
    outlier_x = []

    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            outlier.append(te[i])
            # outlier_x.append(data_x[i])
            outlier_x.append(x[i])
    plt.plot(outlier_x, outlier, 'yo')
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('KPI Value', fontsize=16)
    plt.show()
    for i in range(0, len(te)):
        if (te[i] < threshold1) | (te[i] > threshold2):
            flag = flag + 1
            templ.append(i)
            qlist.append(1)
        else:
            qlist.append(0)
                # print(abs(te[i] - median))
    plt.tick_params(labelsize=12)
    plt.plot(x, qlist)
    plt.xlabel('Time (min)', fontsize=16)
    plt.ylabel('Label Value', fontsize=16)
    plt.show()
    temp=ant.sample_entropy(qlist)
    list1.append(temp)



