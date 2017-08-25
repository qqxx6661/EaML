#!/usr/bin/env python3
# coding=utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time


def _judge_accuracy_ave(predict_array, real_array):

    List_ave = []
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            List_ave.append(100)
            continue
        if predict_array[i] == 0 and real_array[i] == 3:
            List_ave.append(0)
            continue
        if predict_array[i] == 3 and real_array[i] == 0:
            List_ave.append(0)
            continue
        if predict_array[i] == 2 and real_array[i] == 1:
            List_ave.append(0)
            continue
        if predict_array[i] == 1 and real_array[i] == 2:
            List_ave.append(0)
            continue
        List_ave.append(50)
    # print('测试集长度：', len(List_ave))
    # print(List_ave)
    return np.mean(List_ave)


def _judge_accuracy(predict_array, real_array):
    correct = 0
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate


def _train_model_save(x_inner, y_inner, name):
    print('---------', name, '---------')
    print("进行SVM-linear训练")
    start = time.time()
    clf_linear = SVC(kernel='linear').fit(x_inner, y_inner)
    joblib.dump(clf_linear, "model/model_linear_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行SVM-rbf训练")
    start = time.time()
    clf_rbf = SVC().fit(x_inner, y_inner)
    joblib.dump(clf_rbf, "model/model_rbf_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行SVM-sigmoid训练")
    start = time.time()
    clf_sigmoid = SVC(kernel='sigmoid').fit(x_inner, y_inner)
    joblib.dump(clf_sigmoid, "model/model_sigmoid_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行决策树训练")
    start = time.time()
    clf = DecisionTreeClassifier(max_depth=5).fit(x_inner, y_inner)
    joblib.dump(clf, "model/model_clf_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行神经网络训练")
    start = time.time()
    sc = StandardScaler().fit(x_inner)  # 神经网络和逻辑回归需要预处理数据
    x_inner = sc.transform(x_inner)
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500).fit(x_inner, y_inner)
    joblib.dump(mlp, "model/model_mlp_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行逻辑回归训练")
    start = time.time()
    log_reg = linear_model.LogisticRegression(C=1e5).fit(x_inner, y_inner)
    joblib.dump(log_reg, "model/model_logreg_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)


def train_model(train_file_inner, input_frame_number_inner, input_label_delay_inner):
    data = []
    labels_global = []
    labels_upper = []
    max_test_num = 10000
    with open(train_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[2:12]])
            # print(data)
            labels_global.append(tokens[0])
            labels_upper.append(tokens[1])
    print('输入参数个数：', len(data[0]))

    if len(data) > max_test_num:  # 控制最大读取行数
        data = data[-max_test_num:]
        labels_global = labels_global[-max_test_num:]
        labels_upper = labels_upper[-max_test_num:]

    x = np.array(data)
    y_global = np.array(labels_global)
    y_upper = np.array(labels_upper)

    print("读取输入样本数为：", len(x))
    # print(X)
    print("读取标记样本数为：", len(labels_global))
    # print(y)

    _train_model_save(x, y_global, 'global')
    _train_model_save(x, y_upper, 'upper')


def test_accuracy(test_file_inner, input_frame_number, input_label_delay):
    data = []
    labels_global = []
    labels_upper = []

    with open(test_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[2:12]])
            labels_global.append(tokens[0])
            labels_upper.append(tokens[1])

    test_X = np.array(data)
    test_Y_global = np.array(labels_global)
    test_Y_upper = np.array(labels_upper)

    start = time.time()
    clf_linear_global = joblib.load("model/model_linear_global.m")
    test_X_result = clf_linear_global.predict(test_X)
    print("linear全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("linear全局预测准确率2：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)

    start = time.time()
    clf_linear_upper = joblib.load("model/model_linear_global.m")
    test_X_result = clf_linear_upper.predict(test_X)
    print("linear全局预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("linear全局预测准确率2：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)

if __name__ == '__main__':
    test_file = "test/test_2cam_scene2(1)_901.csv"
    train_file = 'data/train_2cam_scene1_01-05.csv'
    input_frame_number = 1  # 输入维度
    input_label_delay = 1  # 预测样本差
    # train_model(train_file, input_frame_number, input_label_delay)
    test_accuracy(train_file, input_frame_number, input_label_delay)