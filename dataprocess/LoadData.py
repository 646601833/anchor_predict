from hdfs.client import Client
import pandas as pd

client = Client("http://10.103.0.11:9870", root='/')

files = []
path = '/lr/pos'
dirt = []
for a, b, c in client.walk(path):
    root = a
    dirt.append(b)
    files = c
col = ['id', 'mmsi', 'latitude', 'longitude', 'course', 'speed', 'lasttm',  'day']
for file in dirt[0]:
    print(file)
    res = []
    with client.read(path+'/'+file+'/'+'part-00000') as read:
        for line in read:
            data = str(line).split('[')[1].split(']')[0]
            l1 = data.split(',')
            l1[2] = float(l1[2])
            l1[3] = float(l1[3])
            l1[4] = float(l1[4])
            l1[5] = float(l1[5])
            l1[6] = int(l1[6])
            res.append(l1)
    df = pd.DataFrame(res, columns=col)
    name = str(file).split('.')[0]
    df.to_excel('../posData/'+name+'.xlsx')

