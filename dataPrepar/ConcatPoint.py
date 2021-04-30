import pandas as pd

df = pd.read_csv('../data/anchor_point2.csv')
df = df.sort_values('lasttm').groupby('id')
data = []
col = ['id', 'lasttm', 'speed', 'latitude', 'longitude', 'day']
for index, value in df:
    value = pd.DataFrame(value)
    data.append(value.loc[value.index[0], :])
res = pd.concat(data, axis=1)
res = res.T
res.to_csv('../data/point2.csv', index=None)
print('finish')
