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
    return correct_rate * 100


def _train_model_save(x_inner, y_inner, name):
    print('---------', name, '---------')
    '''
    print("进行SVM-linear训练")
    start = time.time()
    clf_linear = SVC(kernel='linear').fit(x_inner, y_inner)
    joblib.dump(clf_linear, "model_2cam/model_linear_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)
    '''
    print("进行SVM-rbf训练")
    start = time.time()
    clf_rbf = SVC().fit(x_inner, y_inner)
    joblib.dump(clf_rbf, "model_2cam/model_rbf_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行SVM-sigmoid训练")
    start = time.time()
    clf_sigmoid = SVC(kernel='sigmoid').fit(x_inner, y_inner)
    joblib.dump(clf_sigmoid, "model_2cam/model_sigmoid_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行决策树训练")
    start = time.time()
    clf = DecisionTreeClassifier(max_depth=5).fit(x_inner, y_inner)
    joblib.dump(clf, "model_2cam/model_tree_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行神经网络训练")
    start = time.time()
    sc = StandardScaler().fit(x_inner)  # 神经网络和逻辑回归需要预处理数据
    x_inner = sc.transform(x_inner)
    mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500).fit(x_inner, y_inner)
    joblib.dump(mlp, "model_2cam/model_mlp_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)

    print("进行逻辑回归训练")
    start = time.time()
    log_reg = linear_model.LogisticRegression(C=1e5).fit(x_inner, y_inner)
    joblib.dump(log_reg, "model_2cam/model_logreg_" + name + ".m")
    end = time.time()
    print("执行时间:", end - start)


def train_model(train_file_inner, input_frame_number_inner, input_label_delay_inner):
    data = []
    labels_global = []
    labels_upper = []
    max_test_num = 10000
    delay = input_label_delay_inner

    with open(train_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[2:12]])
            # print(data)
            if delay != 0:  # 推迟label
                delay -= 1
                continue
            labels_global.append(tokens[0])
            labels_upper.append(tokens[1])
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

        labels_global = labels_global[input_frame_number_inner-1:]
        labels_upper = labels_upper[input_frame_number_inner-1:]

    if len(data) > max_test_num:  # 控制最大读取行数
        data = data[-max_test_num:]
        labels_global = labels_global[-max_test_num:]
        labels_upper = labels_upper[-max_test_num:]

    print("输入维度为：", len(data[0]))
    x = np.array(data)
    y_global = np.array(labels_global)
    y_upper = np.array(labels_upper)

    print("读取输入样本数为：", len(x))
    print("读取输出样本数为：", len(labels_global))

    #print('输入：', data)
    #print('输出：', labels_global, labels_upper)

    _train_model_save(x, y_global, 'global')
    _train_model_save(x, y_upper, 'upper')


def cal_accuracy(test_file_inner, input_frame_number_inner, input_label_delay_inner):
    data = []
    labels_global = []
    labels_upper = []
    delay = input_label_delay_inner
    with open(test_file_inner) as file:
        for line in file:
            tokens = line.strip().split(',')
            data.append([tk for tk in tokens[2:12]])
            if delay != 0:  # 推迟label
                delay -= 1
                continue
            labels_global.append(tokens[0])
            labels_upper.append(tokens[1])
    if input_label_delay_inner != 0:
        data = data[:-input_label_delay_inner]  # 删去后面几位
    # print('输入：', data)
    # print('标记：', labels_global, labels_upper)

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

        labels_global = labels_global[input_frame_number_inner-1:]
        labels_upper = labels_upper[input_frame_number_inner-1:]


    test_X = np.array(data)
    test_Y_global = np.array(labels_global)
    test_Y_upper = np.array(labels_upper)

    print("读取输入样本数为：", len(test_X))
    print("读取输出样本数为：", len(test_Y_global))
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
    start = time.time()
    clf_linear_upper = joblib.load("model_2cam/model_linear_upper.m")
    test_X_result = clf_linear_upper.predict(test_X)
    # print("linear上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("linear上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")
    '''
    start = time.time()
    clf_rbf_global = joblib.load("model_2cam/model_rbf_global.m")
    test_X_result = clf_rbf_global.predict(test_X)
    # print("rbf全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("rbf全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    start = time.time()
    clf_rbf_upper = joblib.load("model_2cam/model_rbf_upper.m")
    test_X_result = clf_rbf_upper.predict(test_X)
    # print("rbf上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("rbf上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")

    start = time.time()
    clf_sigmoid_global = joblib.load("model_2cam/model_sigmoid_global.m")
    test_X_result = clf_sigmoid_global.predict(test_X)
    # print("sigmoid全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("sigmoid全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    start = time.time()
    clf_sigmoid_upper = joblib.load("model_2cam/model_sigmoid_upper.m")
    test_X_result = clf_sigmoid_upper.predict(test_X)
    # print("sigmoid上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("sigmoid上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")

    start = time.time()
    clf_tree_global = joblib.load("model_2cam/model_tree_global.m")
    test_X_result = clf_tree_global.predict(test_X)
    # print("tree全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("tree全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    start = time.time()
    clf_tree_upper = joblib.load("model_2cam/model_tree_upper.m")
    test_X_result = clf_tree_upper.predict(test_X)
    # print("tree上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("tree上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")

    # LOC和MLP用
    sc = StandardScaler().fit(test_X)
    test_X = sc.transform(test_X)

    start = time.time()
    clf_logreg_global = joblib.load("model_2cam/model_logreg_global.m")
    test_X_result = clf_logreg_global.predict(test_X)
    # print("logreg全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("logreg全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    start = time.time()
    clf_logreg_upper = joblib.load("model_2cam/model_logreg_upper.m")
    test_X_result = clf_logreg_upper.predict(test_X)
    # print("logreg上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("logreg上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")

    start = time.time()
    clf_mlp_global = joblib.load("model_2cam/model_mlp_global.m")
    test_X_result = clf_mlp_global.predict(test_X)
    # print("mlp全局预测准确率：", _judge_accuracy(test_X_result, test_Y_global))
    print("mlp全局预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_global))
    end = time.time()
    print("执行时间:", end - start)
    start = time.time()
    clf_mlp_upper = joblib.load("model_2cam/model_mlp_upper.m")
    test_X_result = clf_mlp_upper.predict(test_X)
    # print("mlp上层预测准确率：", _judge_accuracy(test_X_result, test_Y_upper))
    print("mlp上层预测准确率：", _judge_accuracy_ave(test_X_result, test_Y_upper))
    end = time.time()
    print("执行时间:", end - start)
    print("-----------")


if __name__ == '__main__':
    test_file = "test/2cam_test_scene1_06.csv"
    train_file = 'data/2cam_train_scene1_01-05.csv'
    input_frame_number = 50  # 输入维度
    input_label_delay = 1  # 预测样本和标签差
    train_model(train_file, input_frame_number, input_label_delay)
    cal_accuracy(train_file, input_frame_number, input_label_delay)
