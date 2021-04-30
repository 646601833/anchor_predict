import os
import random

import numpy as np
from pyalink.alink import *
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from dataprocess.Add_feature import *

pd.set_option('display.max_rows', 1000)


def get_data(dir_name):
    files = os.listdir(dir_name)
    feature = []
    for file in files:
        # print(file)
        path = dir_name + file
        source_df = pd.read_excel(path)
        # 排除没有数据的表格
        if source_df.size == 0:
            print('缺失数据：' + file)
        else:
            # 用0.1替代速度0
            for i in range(source_df['speed'].size):
                if source_df.loc[source_df.index[i], 'speed'] == 0:
                    source_df.loc[source_df.index[i], 'speed'] = 0.1
            # 异常值检测
            y_het = outlier(source_df['speed'].values.reshape(-1, 1), 3, 5)
            # 将异常值用前后数据均值代替
            for j in range(len(y_het)):
                if y_het[j] == -1:
                    source_df.loc[source_df.index[j], 'speed'] = (source_df.loc[source_df.index[j - 1], 'speed'] +
                                                                  source_df.loc[source_df.index[j + 1], 'speed']) / 2
            # 计算每个数据点时间与第一个数据的时间差
            source_df['time'] = source_df['lasttm'].apply(lambda x: x - source_df.loc[source_df.index[0], 'lasttm'])
            # 将每个数据进行hash分桶，间隔为20秒
            source_df['bucket'] = source_df['time'].apply(lambda x: x // 30000)
            # print(source_df[['speed', 'bucket']])
            # 每个桶的数据为平均值
            mean_speed = source_df.groupby('bucket')['speed'].mean()
            speed = np.array(mean_speed.values)
            # 计算速度变化率
            speed_diff = []
            for i in range(speed.size):
                if i > 0:
                    speed_diff.append(np.round((speed[i] - speed[i - 1]) / speed[i - 1], 4))
            # 保证分桶没有缺失，保证数据一致性
            l = list(set(source_df['bucket'].values.tolist()))
            l3 = []
            speed_diff2 = []
            # 缺少20个桶以上的舍弃
            if len(l) >= 40:
                del l[0]
                l3.append(l[0])
                speed_diff2.append(speed_diff[0])
                # 缺失的桶插入前后桶的均值
                for i in range(len(l)):
                    if i > 0:
                        interval = l[i] - l[i - 1]
                        if interval == 1:
                            l3.append(l[i])
                            speed_diff2.append(speed_diff[i])
                        else:
                            for j in range(1, interval):
                                speed_diff2.append((speed_diff[i] + speed_diff[i - 1]) * j / interval)
                                l3.append(l[i - 1] + j)
                            speed_diff2.append(speed_diff[i])
                            l3.append(l[i])
                # 尾部缺少的桶用前面的数据替代
                last = l3[-1]
                if last != 59:
                    for iii in range(59 - l3[-1]):
                        l3.append(last + iii + 1)
                        temp = speed_diff2[-1]
                        speed_diff2.append(temp)
                # 首部缺少的桶用后面的数据替代
                first = l3[0]
                if first != 1:
                    for iii in range(first - 1):
                        l3.insert(0, first - iii - 1)
                        temp = speed_diff2[0]
                        speed_diff2.insert(0, temp)
                # 添加其他属性
                # course_diff = add_course(source_df)
                # max_speed = add_max_speed(source_df)
                # min_speed = add_min_speed(source_df)
                # speed_diff2.append(course_diff)
                # speed_diff2.append(max_speed)
                # speed_diff2.append(min_speed)
                feature.append(speed_diff2)

    return feature


def add_tag(feature, data_type):
    for each in feature:
        if data_type == 'pos':
            each.append(0)
        else:
            each.append(1)
    return feature


def outlier(array, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(array)
    return dbscan.labels_


def split_train_test_data(test_rate, train_rate, start, number):
    pos_data = get_data('../posData/')
    pos_data2 = []
    for each in pos_data:
        pos_data2.append(each[start:start + number])
    pos_data2 = add_tag(pos_data2, 'pos')
    neg_data = get_data('../negData/')
    neg_data2 = []
    for each in neg_data:
        neg_data2.append(each[start:start + number])
    neg_data2 = add_tag(neg_data2, 'neg')
    # 合并正负数据
    for each in neg_data2:
        pos_data2.append(each)
    # 分离出标签
    data_y = []
    for each in pos_data2:
        y = each.pop()
        data_y.append(y)
    X_train, X_test, y_train, y_test = train_test_split(pos_data2, data_y, test_size=test_rate, train_size=train_rate,
                                                        random_state=random.randint(1, 1000))
    return X_train, X_test, y_train, y_test


def linear_log_reg(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    pre_test = log_reg.predict(X_test)
    # # 准确率与召回率
    acu = accuracy_score(y_test, pre_test)
    rec = recall_score(y_test, pre_test, average='macro')
    return acu, rec


def no_linear_log_reg(X_train, X_test, y_train, y_test):
    # 非线性逻辑回归
    poly_log = PolynomialFeatures(degree=2)
    x_poly = poly_log.fit_transform(X_train)
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(x_poly, y_train)
    pre_test = log_reg.predict(poly_log.fit_transform(X_test))
    # # 准确率与召回率
    acu = accuracy_score(y_test, pre_test)
    rec = recall_score(y_test, pre_test, average='macro')
    return acu, rec


def linear_svc(X_train, X_test, y_train, y_test):
    svc = LinearSVC(loss='hinge')
    svc.fit(X_train, y_train)
    pre_test = svc.predict(X_test)
    acu = accuracy_score(y_test, pre_test)
    rec = recall_score(y_test, pre_test, average='macro')
    return acu, rec


def no_linear_svc(X_train, X_test, y_train, y_test):
    svc = LinearSVC(C=1, loss="hinge", random_state=42)
    poly_log = PolynomialFeatures(degree=2)
    x_poly = poly_log.fit_transform(X_train)
    svc.fit(x_poly, y_train)
    pre_test = svc.predict(poly_log.fit_transform(X_test))
    acu = accuracy_score(y_test, pre_test)
    rec = recall_score(y_test, pre_test, average='macro')
    return acu, rec


def learn_curve(x, y, model, no_linear):
    if no_linear:
        poly_log = PolynomialFeatures(degree=2)
        x = poly_log.fit_transform(x)
    x_shuffle, y_shuffle = shuffle(x, y)
    train_sizes, train_scores, valid_scores = learning_curve(model, x_shuffle, y_shuffle,
                                                             train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0], cv=10,
                                                             scoring='accuracy', n_jobs=-1)
    train_acc = np.mean(train_scores, axis=1)
    test_acc = np.mean(valid_scores, axis=1)
    return train_acc, test_acc, train_sizes
    # plt.plot(train_sizes, train_loss_mean, 'o-', color="r",
    #          label="Training")
    # plt.plot(train_sizes, test_loss_mean, 'o-', color="g",
    #          label="Cross-validation")
    #
    # plt.xlabel("Training examples")
    # plt.ylabel("accuracy")
    # plt.title(model_name)
    # plt.legend(loc="best")
    # plt.show()


def get_x_y():
    pos_data = get_data('../posData/')
    y = []
    x = []
    for i in range(len(pos_data)):
        y.append(0)
        x.append(pos_data[i])
    neg_data = get_data('../negData/')
    for i in range(len(neg_data)):
        y.append(1)
        x.append(neg_data[i])
    return x, y


def cross_validation(x, y, model, nolinear):
    if nolinear:
        poly_log = PolynomialFeatures(degree=2)
        x = poly_log.fit_transform(x)
    x_shuffle, y_shuffle = shuffle(x, y, random_state=random.randint(1, 1000))
    acc = cross_val_score(model, x_shuffle, y_shuffle, scoring='accuracy', cv=10, n_jobs=-1)
    acc = np.mean(acc, axis=0)
    rec = cross_val_score(model, x_shuffle, y_shuffle, scoring='recall', cv=10, n_jobs=-1)
    rec = np.mean(rec, axis=0)
    return acc, rec


def alink_pre():
    x, y = get_x_y()
    poly_log = PolynomialFeatures(degree=2)
    x = poly_log.fit_transform(x)
    df = pd.DataFrame(x)
    df['label'] = y
    schema = ''
    for i in range(len(x[0])):
        schema += 'f{} double,'.format(i)
    schema += 'label int'
    data = dataframeToOperator(df, schema, op_type='batch')
    data_shuffled = data.link(ShuffleBatchOp())
    # 数据拆分
    spliter = SplitBatchOp().setFraction(0.8).setRandomSeed(667).linkFrom(data_shuffled)
    train_data = spliter
    test_data = spliter.getSideOutput(0)
    col = ['f{}'.format(i) for i in range(len(x[0]))]
    # lr = LogisticRegressionTrainBatchOp().setFeatureCols(col).setLabelCol('label').setMaxIter(1000)
    svm = LinearSvmTrainBatchOp().setFeatureCols(col).setLabelCol('label').setMaxIter(1000)
    model = train_data.link(svm)
    reserved_col = ['label', 'pred', 'details']
    predictor = LinearSvmPredictBatchOp().setPredictionCol('pred').setPredictionDetailCol('details').setReservedCols(
        reserved_col).linkFrom(model, test_data)
    # predictor = LogisticRegressionPredictBatchOp() \
    #     .setPredictionCol('pred') \
    #     .setPredictionDetailCol('details') \
    #     .setReservedCols(reserved_col) \
    #     .linkFrom(model, test_data)
    predictor.print()
    metrics = EvalBinaryClassBatchOp().setLabelCol('label').setPredictionDetailCol('details').linkFrom(
        predictor).collectMetrics()
    return metrics.getAccuracy(), metrics.getMicroRecall()


if __name__ == '__main__':
    acu_list = []
    rec_list = []
    acu_list2 = []
    rec_list2 = []
    acu_list3 = []
    rec_list3 = []
    acu_list4 = []
    rec_list4 = []
    for i in range(0, 20):
        X_train, X_test, y_train, y_test = split_train_test_data(0.1, 0.9, i, 40)
        acu, rec = no_linear_svc(X_train, X_test, y_train, y_test)
        acu_list.append(acu)
        rec_list.append(rec)
        acu2, rec2 = linear_svc(X_train, X_test, y_train, y_test)
        acu_list2.append(acu2)
        rec_list2.append(rec2)
        acu3, rec3 = no_linear_log_reg(X_train, X_test, y_train, y_test)
        acu_list3.append(acu3)
        rec_list3.append(rec3)
        acu4, rec4 = linear_log_reg(X_train, X_test, y_train, y_test)
        acu_list4.append(acu4)
        rec_list4.append(rec4)
    index = [i for i in range(40, 60)]

    ax1 = plt.subplot(221)
    plt.plot(index, acu_list, 'r', label='accuracy')
    plt.plot(index, rec_list, 'g-.', label='recall')
    plt.title('no_linear_svc')
    plt.legend()

    ax2 = plt.subplot(222)
    plt.plot(index, acu_list2, 'r', label='accuracy')
    plt.plot(index, rec_list2, 'g-.', label='recall')
    plt.title('linear_svc')
    plt.legend()

    ax3 = plt.subplot(223)
    plt.plot(index, acu_list3, 'r', label='accuracy')
    plt.plot(index, rec_list3, 'g-.', label='recall')
    plt.title('no_linear_log_reg')
    plt.legend()

    ax4 = plt.subplot(224)
    plt.plot(index, acu_list4, 'r', label='accuracy')
    plt.plot(index, rec_list4, 'g-.', label='recall')
    plt.title('linear_log_reg')
    plt.legend()
    plt.show()

    # # 保存模型
    # joblib.dump(log_reg, '../model/poly_log_reg.pkl')
    # print(log_reg.score(poly_log.fit_transform(X_train), y_train))
    # print(log_reg.score(poly_log.fit_transform(X_test), y_test))
