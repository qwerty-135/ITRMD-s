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
boxes3 = read_pickle("machine-1-1_test.pkl")
boxes3=np.squeeze(np.array(boxes3),axis=0)
list=[]
for i in range(0,38):
   X = boxes3[:, i]
   x1 = np.linspace(1, 28479, num=28479)
   x2 = np.linspace(1, 475, num=475)

   n_timestamps = len(X)  # 时间戳长度就是有多少个数字
   window_size = 60  # 定义窗口大小为2
   transformer = PiecewiseAggregateApproximation(window_size=window_size)  # 实例化
   X_transform = transformer.transform(X.reshape(1, -1))  # 转换
   list.append(X_transform[0])
   plt.plot(x1,X,label='Original')
   plt.plot(np.arange(window_size // 2,n_timestamps + window_size // 2,window_size),X_transform[0],'o--',label='PAA')
   plt.legend(loc='best', fontsize=10) #loc='best'自动寻找最好的位置
   plt.xlabel('Time (min)', fontsize=16)
   plt.ylabel('KPI Value', fontsize=16)
   plt.savefig('savefig_example.png')
   plt.show()

np.save('test0.npy',list)