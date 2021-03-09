# coding: utf-8
import csv
import time

import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle

import dimension_reduce as dr
import imbalance_strategies as imb
import model_measure_functions as mf
from scipy import sparse

    
def get_classifier(clf_name):
    if clf_name == 'lr':
        clf = LogisticRegression()
    elif clf_name == 'svm':
        # the kernel can also be 'linear', 'rbf','polynomial','sigmoid', etc.
        clf = svm.SVC(kernel='linear', probability=True)
    #        clf = svm.SVC(kernel='rbf', probability = True, class_weight = 1, decision_function_shaope = 'ovr')
    elif clf_name == 'mlp':
        clf = MLPClassifier(max_iter=500,shuffle = True)
    elif clf_name == 'nb':
        clf = MultinomialNB()
    elif clf_name == 'rf':
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
    else:
        print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
    return clf

#data_name = 'ambari'  # the csv file name of input folder
#['ambari','camel','derby','openstack','chromium','wicket']
#['hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec','zookeeper_sec','flume_sec']
    
# ['hbase_perf','hdfs_perf','cassandra_perf','mapreduce_perf','zookeeper_perf','flume_perf']
#['ambari_perf','camel_perf','derby_perf','wicket_perf']
data_names =['openstack']
clf_names = ['rf'] #,'nb','lr','svm','mlp','rf'

for data_name in data_names:
    print("begin: ", data_name)
    input_file0 = "../input_sec/" + data_name + "_0.csv"
    input_file1 = "../input_sec/" + data_name + "_1.csv"
    sourcedata0 = pandas.read_csv(input_file0).fillna('')
    sourcedata1 = pandas.read_csv(input_file1).fillna('')
    
    num0 = int(1/10*len(sourcedata0))
    num1 = int(1/10*len(sourcedata1))
    
    for clf_name in clf_names:
        clf = get_classifier(clf_name)
        kfold_output = '../output_sec/baseline/bl_' + data_name + \
                               '_' + clf_name +'_output_smt.csv'        
                            
        csv_file_k = open(kfold_output, "w", newline='')
        writer_o = csv.writer(csv_file_k, delimiter=',')
        writer_o.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'FPR', 'Precision', 'f_measure',
                     'g_measure', 'mcc', 'accuracy', 'auc', 'PofB20', 'opt'])
        for j in range(0,10):
            test_data = np.vstack((sourcedata0[j*num0:(j+1)*num0], sourcedata1[j*num1:(j+1)*num1]))
            sourcedata0_left = sourcedata0.drop(sourcedata0.index[j*num0:(j+1)*num0])
            sourcedata1_left = sourcedata1.drop(sourcedata1.index[j*num1:(j+1)*num1])    
            train_data = np.vstack((sourcedata0_left, sourcedata1_left))
        
            train_data = shuffle(train_data)
            train_content = train_data[:,0].tolist()
            train_label = train_data[:,1].tolist()  
            
            test_data = shuffle(test_data)            
            test_content = test_data[:,0].tolist()
            test_label = test_data[:,1].tolist()
            
            vectorizer = CountVectorizer(stop_words='english')        
    
            train_content_matrix = vectorizer.fit_transform(train_content)        
            test_content_matrix = vectorizer.transform(test_content)   
            data_content_matrix_smt, data_label_smt = imb.get_ovs_smote_standard(train_content_matrix, train_label)   
#            data_content_matrix_smt, data_label_smt = imb.get_uds_rdm(train_content_matrix, train_label)         
            clf.fit(data_content_matrix_smt, data_label_smt)
            predicted = clf.predict(test_content_matrix)
            predicted_proba = clf.predict_proba(test_content_matrix)    
            TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, mcc, accuracy, auc, PofB20, opt \
                    = mf.model_measure_with_cross(predicted, predicted_proba, test_label)
            writer_o.writerow([len(train_label),TP, FN, TN, FP, pd, pf, prec, f_measure,
                                 g_measure, mcc, accuracy, auc, PofB20, opt])  
            print(len(train_label),TP, FN, TN, FP, pd, pf, prec, f_measure,
                             g_measure, mcc, accuracy, auc, PofB20, opt)    
        csv_file_k.close()
    print('classifier finished:', clf_name)
print("All finished for dataset: ", data_name)
