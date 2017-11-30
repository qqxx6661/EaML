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


def _judge_accuracy(predict_array, real_array):
    correct = 0
    for i in range(len(predict_array)):
        if predict_array[i] == real_array[i]:
            # print(predict_array[i], real_array[i])
            correct += 1
    correct_rate = correct / len(predict_array)
    return correct_rate * 100


def _train_model_save(x_inner, y_inner, name):
    print('---------', name, '---------')
    '''
    print("进行SVM-linear训练")
    start = time.time()
    clf_linear = SVC(kernel='linear').fit(x_inner, y_inner)
    joblib.dump(clf_linear, "ML_model/model_linear_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)
    '''
    print("进行SVM-rbf训练")
    start = time.time()
    clf_rbf = SVC().fit(x_inner, y_inner)
    joblib.dump(clf_rbf, "ML_model/model_rbf_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行SVM-sigmoid训练")
    start = time.time()
    clf_sigmoid = SVC(kernel='sigmoid').fit(x_inner, y_inner)
    joblib.dump(clf_sigmoid, "ML_model/model_sigmoid_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行决策树训练")
    start = time.time()
    clf = DecisionTreeClassifier(max_depth=5).fit(x_inner, y_inner)
    joblib.dump(clf, "ML_model/model_tree_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行神经网络训练")
    start = time.time()
    sc = StandardScaler().fit(x_inner)  # 神经网络和逻辑回归需要预处理数据
    x_inner = sc.transform(x_inner)
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500).fit(x_inner, y_inner)
    joblib.dump(mlp, "ML_model/model_mlp_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行逻辑回归训练")
    start = time.time()
    log_reg = linear_model.LogisticRegression(C=1e5).fit(x_inner, y_inner)
    joblib.dump(log_reg, "ML_model/model_logreg_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)
    print('-----------------')


def train_model(train_file_inner, input_frame_number_inner, input_label_delay_inner):
    data = []
    labels = []
    max_train_num = 10000
    delay = input_label_delay_inner

    with open(train_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[1:8]])
            if delay != 0:  # 推迟label
                delay -= 1
                continue
            labels.append(tokens[0])
    if input_label_delay_inner:
        data = data[:-input_label_delay_inner]  # 删去后面几位

    if input_frame_number_inner != 1:
        delay_vector = input_frame_number_inner
        temp_vector = []
        temp_data = []
        # 由于上面已经延迟，所以每个输入对应的输出是输入的最后一行后面的标签
        for line_idx in range(len(data)-input_frame_number_inner+1):
            temp_idx = line_idx
            while delay_vector:
                temp_vector += data[temp_idx]
                # print('临时为：', temp_vector)
                temp_idx += 1
                delay_vector -= 1

            delay_vector = input_frame_number_inner
            temp_data.append(temp_vector)
            temp_vector = []
        data = temp_data

        labels = labels[input_frame_number_inner-1:]

    if len(data) > max_train_num:  # 控制最大读取行数
        data = data[-max_train_num:]
        labels = labels[-max_train_num:]

    print("输入维度为：", len(data[0]))
    x = np.array(data)
    y = np.array(labels)

    print("读取输入样本数为：", len(x))
    print("读取输出样本数为：", len(labels))

    for i, line in enumerate(data):
        print(line, labels[i])

    _train_model_save(x, y, 'global')


def cal_accuracy(test_file_inner, input_frame_number_inner, input_label_delay_inner):
    data = []
    labels = []
    delay = input_label_delay_inner
    with open(test_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[1:8]])
            if delay != 0:  # 推迟label
                delay -= 1
                continue
            labels.append(tokens[0])
    if input_label_delay_inner != 0:
        data = data[:-input_label_delay_inner]  # 删去后面几位

    if input_frame_number_inner != 1:
        delay_vector = input_frame_number_inner
        temp_vector = []
        temp_data = []
        # 由于上面已经延迟，所以每个输入对应的输出是输入的最后一行后面的标签
        for line_idx in range(len(data)-input_frame_number_inner+1):
            temp_idx = line_idx
            while delay_vector:
                temp_vector += data[temp_idx]
                # print('临时为：', temp_vector)
                temp_idx += 1
                delay_vector -= 1

            delay_vector = input_frame_number_inner
            temp_data.append(temp_vector)
            temp_vector = []
        data = temp_data

        labels = labels[input_frame_number_inner-1:]

    test_X = np.array(data)
    test_Y = np.array(labels)

    print("读取输入样本数为：", len(test_X))
    print("读取输出样本数为：", len(test_Y))
    # print('输入：', data)
    # print('输出：', labels_global, labels_upper)

    '''
    start = time.time()
    clf_linear_global = joblib.load("model_2cam/model_linear_global.m")
    test_X_result = clf_linear_global.predict(test_X)
    # print("linear全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("linear全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    '''
    start = time.time()
    clf_rbf_global = joblib.load("ML_model/model_rbf_global.m")
    test_X_result = clf_rbf_global.predict(test_X)
    # print(test_X_result)
    # print(test_Y)
    print("rbf全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
    end = time.time()
    print("执行时间:", end - start)


    start = time.time()
    clf_sigmoid_global = joblib.load("ML_model/model_sigmoid_global.m")
    test_X_result = clf_sigmoid_global.predict(test_X)
    print("sigmoid全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
    end = time.time()
    print("执行时间:", end - start)


    start = time.time()
    clf_tree_global = joblib.load("ML_model/model_tree_global.m")
    test_X_result = clf_tree_global.predict(test_X)
    print(test_X_result)
    # print(test_Y)
    print("tree全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
    end = time.time()
    print("执行时间:", end - start)


    # LOC和MLP用
    sc = StandardScaler().fit(test_X)
    test_X = sc.transform(test_X)

    start = time.time()
    clf_logreg_global = joblib.load("ML_model/model_logreg_global.m")
    test_X_result = clf_logreg_global.predict(test_X)
    print("logreg全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
    end = time.time()
    print("执行时间:", end - start)

    start = time.time()
    clf_mlp_global = joblib.load("ML_model/model_mlp_global.m")
    test_X_result = clf_mlp_global.predict(test_X)
    print("mlp全局预测准确率：", _judge_accuracy(test_X_result, test_Y))
    end = time.time()
    print("执行时间:", end - start)


if __name__ == '__main__':
    test_file = "gallery/test/yolo_1607_person_0_ML.csv"
    train_file = 'gallery/train/yolo_1547_person_0_ML.csv'
    input_frame_number = 30  # 输入维度
    input_label_delay = 10 # 预测样本和标签差
    train_model(train_file, input_frame_number, input_label_delay)
    cal_accuracy(test_file, input_frame_number, input_label_delay)

