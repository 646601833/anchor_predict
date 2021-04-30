import math

import pandas as pd


def add_course(df):
    a = df['course'].diff(1).apply(lambda x: math.fabs(x)).sum()
    return a


def add_max_speed(df):
    max = df['speed'].max()
    return max


def add_min_speed(df):
    min = df['speed'].min()
    return min


if __name__ == '__main__':
    df = pd.read_excel('../data2/query-hive-294.csv')
    # course = add_course(df)
    # print(course)
    max = add_max_speed(df)
    print(max)