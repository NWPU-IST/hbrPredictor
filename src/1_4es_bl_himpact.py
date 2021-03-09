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
data_name ='Ambari_com'
clf_name = 'nb' #,'nb','lr','svm','mlp','rf'

print(data_name, " Begin ",  clf_name)
input_file = "../input/" + data_name + ".csv"
sourcedata = pandas.read_csv(input_file).fillna('')
#sourcedata = sourcedata.sort_values(by="bugid", ascending=False)
#(key=lambda a: a["key"])
content = sourcedata.description
label = sourcedata.com.tolist()
label_sec = sourcedata.Security.tolist()
#label_perf = sourcedata.perf.tolist()
#label_config = sourcedata.config.tolist()

#print(label)
#print(label_sec)
#print(label_perf)
#print(label_config)
num0 = int(1/11*len(label))
#print(len(content), len(label), num0)
clf = get_classifier(clf_name)    

for m in range(0,10):
    kfold_output = '../output_tmp/' + data_name + \
                           '_' + clf_name +'_output_com1_'+str(m)+'.csv'        
                        
    csv_file_k = open(kfold_output, "w", newline='')
    writer_o = csv.writer(csv_file_k, delimiter=',')
    writer_o.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'Precision', 'F_measure', 'success_rate'])
    for j in range(0,10):
#        print('**********this is j: ',j)

        train_content = content[j*num0:(j+1)*num0]      
        train_label = label[j*num0:(j+1)*num0]
        if j<9:
#            print("j < 4")
            test_content = content[(j+1)*num0:(j+2)*num0]  
            test_label = label[(j+1)*num0:(j+2)*num0] 
            test_label_sec = label_sec[(j+1)*num0:(j+2)*num0] 
#            test_label_perf = label_perf[(j+1)*num0:(j+2)*num0] 
#            test_label_config = label_config[(j+1)*num0:(j+2)*num0] 
        else:
#            print("j==4")
            test_content = content[(j+1)*num0:]  
            test_label = label[(j+1)*num0:] 
            test_label_sec = label_sec[(j+1)*num0:] 
#            test_label_perf = label_perf[(j+1)*num0:] 
#            test_label_config = label_config[(j+1)*num0:] 
#            print(test_content)
            
        vectorizer = CountVectorizer(stop_words='english')        

        train_content_matrix = vectorizer.fit_transform(train_content)        
        test_content_matrix = vectorizer.transform(test_content)   
#        data_content_matrix_smt, data_label_smt = imb.get_ovs_smote_standard(train_content_matrix, train_label)   
#        data_content_matrix_smt, data_label_smt = imb.get_uds_rdm(train_content_matrix, train_label)         
        clf.fit(train_content_matrix, train_label)
        predicted = clf.predict(test_content_matrix)  
#        print(predicted,test_label)
#        TP, FN, TN, FP, pd, prec, f_measure, auc \
#                = mf.model_measure_f1_auc(predicted, test_label)
        TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, success_rate \
                = mf.model_measure_basic(predicted, test_label)
        writer_o.writerow([len(train_label),TP, FN, TN, FP, pd, prec, f_measure, success_rate])  
        print("com performance: ",len(train_label),TP, FN, TN, FP, pd, prec, f_measure,success_rate)   

        TP_sec, FN_sec, TN_sec, FP_sec, pd_sec, pf_sec, prec_sec, f_measure_sec, g_measure_sec, success_rate_sec \
                = mf.model_measure_basic(predicted, test_label_sec)        
        print("sec performance: ",len(train_label),TP_sec, FN_sec, TN_sec, FP_sec, pd_sec, pf_sec, prec_sec, f_measure_sec, success_rate_sec)   

#        TP_perf, FN_perf, TN_perf, FP_perf, pd_perf, pf_perf, prec_perf, f_measure_perf, g_measure_perf, success_rate_perf \
#                = mf.model_measure_basic(predicted, test_label_perf)        
#        print("perf performance: ",len(train_label),TP_perf, FN_perf, TN_perf, FP_perf, pd_perf, pf_perf, prec_perf, f_measure_perf, success_rate_perf)   
#
#        TP_config, FN_config, TN_config, FP_config, pd_config, pf_config, prec_config, f_measure_config, g_measure_config, success_rate_config \
#                = mf.model_measure_basic(predicted, test_label_config)        
#        print("config performance: ",len(train_label),TP_config, FN_config, TN_config, FP_config, pd_config, pf_config, prec_config, f_measure_config, success_rate_config)   

    print(m)
    csv_file_k.close()
print("All finished for dataset: ", data_name)
