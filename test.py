from hdfs.client import Client
import os
from sklearn.preprocessing import scale
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC
import time
import numpy as np

t1 = time.time()
for i in np.linspace(0, 10000, 10000):
    print(i)
t2 = time.time()
print(t2 - t1)