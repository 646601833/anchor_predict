import numpy as np
import pandas as pd


if __name__ == '__main__':
    data = []
    source_data = pd.read_csv('../data/anchor_neg.csv')
    df = source_data.sort_values('lasttm').groupby('id')
    a = 0
    for index,value in df:
        a += 1
        value = pd.DataFrame(value)
        temp = 0
        for i in range(value['speed'].size):
            if value.loc[value.index[i], 'speed'] <= 1:
                temp += 1
        if temp/value['speed'].size < 0.8:
            data.append(value)
            # print(index)
            # print(value[['lasttm', 'speed']])

    print(a)
    print(len(data))