import tensorflow as tf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
import matplotlib.patches as mpatches
kmeans = KMeans(n_clusters=3, n_init=20)
list=[]
for index in range(38):

    boxes = np.load('true.npy')


    boxes = boxes[:, -1, index]
    # print(len(boxes))
    boxes = np.ravel(boxes)
    # print(len(boxes))
    list.append(boxes)
list=np.squeeze(np.array(list))
print(len(list[0]))

box=np.load("test0.npy")

model = MeanShift()

yhat = model.fit_predict(box)

print(yhat)
