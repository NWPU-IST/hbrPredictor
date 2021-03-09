# -*- coding: utf-8 -*-
__author__ = 'Junzheng Chen'

from sklearn import metrics
import cmath


# clf是输入的分类器
# test_content_count 是输入的测试集样本
# test_label 是输入的测试集标签
def calc_auc(clf, test_content_count, test_label):
    # fpr=false positive rate
    # tpr=true positive rate
    # pos_label: label stands for positive,我们认为1，即有安全性bug是正面的
    pred = clf.predict(test_content_count)
    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    return metrics.auc(fpr, tpr)


# 变量含义同上
def calc_PofB20(clf, test_content_count, test_label):
    proba_result = clf.predict_proba(test_content_count)
    # 定义了一个数据结构，为[标签0的可能性，标签1的可能性，编号]
    new_result = []
    i = 0
    for line in proba_result.tolist():
        line.append(i)
        i += 1
        new_result.append(line)
    # 对所有测试样本为安全性bug的可能性从大到小排序
    # key是排序规则，这里我们是用为1的可能性进行排序
    # reverse为True代表降序
    sort_new_result = sorted(new_result, key=lambda line: line[1], reverse=True)
    # 所有含有安全性bug的编号列表
    security_bug_list = []
    for i in range(0, len(test_label)):
        if test_label[i] == 1:
            security_bug_list.append(i)
    # 设置20%对应的数量
    percent_20_length = int(0.2 * len(test_label))
    correct_number = 0
    for i in range(0, percent_20_length):
        if sort_new_result[i][-1] in security_bug_list:
            correct_number += 1
    return float(correct_number / len(security_bug_list))


# 变量含义同
def calc_opt(clf, test_content_count, test_label):
    proba_result = clf.predict_proba(test_content_count)
    # 定义了一个数据结构，为[标签0的可能性，标签1的可能性，编号]
    new_result = []
    i = 0
    for line in proba_result.tolist():
        line.append(i)
        i += 1
        new_result.append(line)
    # 对所有测试样本为安全性bug的可能性从大到小排序
    # key是排序规则，这里我们是用为1的可能性进行排序
    # reverse为True代表降序
    sort_new_result = sorted(new_result, key=lambda line: line[1], reverse=True)
    # 相差总数，经过处理后近似成为面积
    sum = 0
    # 含有的为1的总数
    security_bug_length = test_label.count(1)
    security_bug_list = []
    for i in range(0, len(test_label)):
        if test_label[i] == 1:
            security_bug_list.append(i)
    # 累计的正确数
    correct_num = 0
    for i in range(0, len(test_label)):
        optimal = security_bug_length if (i + 1) > security_bug_length else i + 1
        correct_num += (1 if sort_new_result[i][-1] in security_bug_list else 0)
        predict = correct_num
        opt_temp = optimal - predict
        sum += opt_temp
    opt = float(sum / security_bug_length / len(test_label))
    return 1 - opt

def model_measure_basic(pred, test_label):
    # 初始化一些变量
    init_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TP, FN, TN, FP, pd, pf, prec, f_measure,g_measure,success_rate = init_index
    # 计算TP FN TN FP
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            FN += 1
        elif pred[i] == test_label[i] == 0:
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            FP += 1
    # 计算pd pf prec f_measure g_measure
    if TP + FN != 0:
        pd = TP / (TP + FN)
    if FP + TN != 0:
        pf = FP / (FP + TN)
    if TP + FP != 0:
        prec = TP / (TP + FP)
    if pd + prec != 0:
        f_measure = 2 * pd * prec / (pd + prec)
    g_measure = (2 * pd * (1 - pf)) / (pd + (1 - pf))
    if TP + TN + FN + FP !=0:
        success_rate = float((TP + TN) / (TP + TN + FN + FP))


    return TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate

def model_measure_f1_auc(pred, test_label):
    pred = pred.tolist()
    init_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TP, FN, TN, FP, pd, fpr, prec, f_measure, g_measure, mcc = init_index
    # 计算TP FN TN FP
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            FN += 1
        elif pred[i] == test_label[i] == 0:
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            FP += 1
    # 计算pd pf prec f_measure g_measure
    if TP + FN != 0:
        pd = round(TP / (TP + FN), 4)

    if TP + FP != 0:
        prec = round(TP / (TP + FP), 4)
    if pd + prec != 0:
        f_measure = round(2 * pd * prec / (pd + prec), 4)

    fpr, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    auc_tmp = metrics.auc(fpr, tpr)
    auc = round(auc_tmp, 4)

    return TP, FN, TN, FP, pd, prec, f_measure, auc

# 返回值说明 返回值为TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate, auc, PofB20, opt
def model_measure_with_cross(pred, pred_proba, test_label):
    pred = pred.tolist()
    init_index = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TP, FN, TN, FP, pd, fpr, prec, f_measure, g_measure, mcc = init_index
    # 计算TP FN TN FP
    for i in range(len(test_label)):
        if pred[i] == test_label[i] == 1:
            TP += 1
        elif test_label[i] == 1 and pred[i] != test_label[i]:
            FN += 1
        elif pred[i] == test_label[i] == 0:
            TN += 1
        elif test_label[i] == 0 and pred[i] != test_label[i]:
            FP += 1
    # 计算pd pf prec f_measure g_measure
    if TP + FN != 0:
        pd = round(TP / (TP + FN), 4)
    if FP + TN != 0:
        fpr = round(FP / (FP + TN), 4)
    if TP + FP != 0:
        prec = round(TP / (TP + FP), 4)
    if pd + prec != 0:
        f_measure = round(2 * pd * prec / (pd + prec), 4)
    if fpr != 1:
        g_measure = round((2 * pd * (1 - fpr)) / (pd + (1 - fpr)), 4)
    if (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) != 0:
        mcc_tmp = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))**0.5
        mcc = round(mcc_tmp, 4)
    if (TP + TN + FN + FP) != 0:
        accuracy = round((TP + TN) / (TP + TN + FN + FP), 4)

    fpr_1, tpr, thresholds = metrics.roc_curve(test_label, pred, pos_label=1)
    auc_tmp = metrics.auc(fpr_1, tpr)
    auc = round(auc_tmp, 4)

    # 概率计算部分
    proba_result = pred_proba
    # 定义了一个数据结构，为[标签0的可能性，标签1的可能性，编号]
    new_result = []
    i = 0
    for line in proba_result.tolist():
        line.append(i)
        i += 1
        new_result.append(line)
    # 对所有测试样本为安全性bug的可能性从大到小排序
    # key是排序规则，这里我们是用为1的可能性进行排序
    # reverse为True代表降序
    sort_new_result = sorted(new_result, key=lambda line: line[1], reverse=True)
    # 所有含有安全性bug的编号列表
    security_bug_list = []
    for i in range(0, len(test_label)):
        if test_label[i] == 1:
            security_bug_list.append(i)
    # 设置20%对应的数量
    percent_20_length = int(0.2 * len(test_label))
    correct_number = 0
    for i in range(0, percent_20_length):
        if sort_new_result[i][-1] in security_bug_list:
            correct_number += 1
    PofB20_tmp = float(correct_number / len(security_bug_list))

    sum = 0
    security_bug_length = len(security_bug_list)
    # 累计的正确数
    correct_num = 0
    for i in range(0, len(test_label)):
        optimal = security_bug_length if (i + 1) > security_bug_length else i + 1
        correct_num += (1 if sort_new_result[i][-1] in security_bug_list else 0)
        predict = correct_num
        opt_temp = optimal - predict
        sum += opt_temp
    opt_tmp = float(sum / security_bug_length / len(test_label))
    
    PofB20 = round(PofB20_tmp,4)
    opt = round(opt_tmp,4)

    return TP, FN, TN, FP, pd, fpr, prec, f_measure, g_measure, mcc, accuracy, auc, PofB20, 1 - opt
