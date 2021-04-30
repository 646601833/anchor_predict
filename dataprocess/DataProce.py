import numpy as np
import pandas as pd

if __name__ == '__main__':
    source_data = pd.read_csv('../data/anchor_track.csv')
    df = source_data.sort_values('lasttm').groupby('id')
    # data = []
    for index, value in df:
        value = pd.DataFrame(value)
        if value.shape[0] >= 50:
            for i in range(value['speed'].size):
                if i == 0:
                    value.loc[value.index[i], 'speed_diff'] = np.NaN
                else:
                    value.loc[value.index[i], 'speed_diff'] = (value.loc[value.index[i], 'speed'] - value.loc[
                        value.index[i - 1], 'speed']) / value.loc[value.index[i - 1], 'speed']
            value.drop(value.index[0], inplace=True)
            value['std'] = value['speed_diff'].std()
            print(index)
            print(value[['lasttm', 'speed', 'speed_diff', 'std', 'latitude', 'longitude']])
