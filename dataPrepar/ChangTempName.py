import pandas as pd
from hdfs.client import Client

client = Client("http://10.103.0.11:9870", root='/')
path = '/user/hive/warehouse/ods.db/anchor_track'

l = client.list(path)
for each in l:
    if str(each).startswith('.'):
        name = str(each).split('.')[1]
        client.rename(path+'/'+str(each), path+'/'+str(name))
print('finish')
