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
from sklearn.neighbors import KNeighborsClassifier

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
        clf = MLPClassifier(max_iter=5000,shuffle = True)
    elif clf_name == 'nb':
        clf = MultinomialNB()
    elif clf_name == 'rf':
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
    elif clf_name == 'knn':
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
    return clf

data_name = 'cassandra_sec'  # the csv file name of input folder
clf_names = ['nb'] #,'nb','lr','svm','mlp','rf'
thr = 100
#['ambari','camel','derby','openstack','chromium','wicket']
#['hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec','zookeeper_sec','flume_sec']
    
# ['hbase_perf','hdfs_perf','cassandra_perf','mapreduce_perf','zookeeper_perf','flume_perf']
#['ambari_perf','camel_perf','derby_perf','wicket_perf']

#train_data, cand_data, validate_data, test_data = split_1_7_1_1(data_name)

input_file0 = "../input_sec/" + data_name + "_0.csv"
input_file1 = "../input_sec/" + data_name + "_1.csv"
sourcedata0 = pandas.read_csv(input_file0).fillna('')
sourcedata0 = shuffle(sourcedata0)
sourcedata1 = pandas.read_csv(input_file1).fillna('')
sourcedata1 = shuffle(sourcedata1)

num0 = int(1/10*len(sourcedata0))
num1 = int(1/10*len(sourcedata1))
print("begin dataset: ", data_name)
for clf_name in clf_names:
    print("Begin classifier: ", clf_name)
    clf = get_classifier(clf_name)
    kfold_output = '../output_sec/iml_al/by_clf_name/' + data_name + \
                           '_' + clf_name +'_output_1.csv'        
                        
    csv_file_k = open(kfold_output, "w", newline='')
    writer_o = csv.writer(csv_file_k, delimiter=',')
    writer_o.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'FPR', 'Precision', 'f_measure',
                 'g_measure', 'mcc', 'accuracy', 'auc', 'PofB20', 'opt'])
    for j in range(0,10):
        test_data = np.vstack((sourcedata0[j*num0:(j+1)*num0], sourcedata1[j*num1:(j+1)*num1]))
        sourcedata0_left = sourcedata0.drop(sourcedata0.index[j*num0:(j+1)*num0])
        sourcedata1_left = sourcedata1.drop(sourcedata1.index[j*num1:(j+1)*num1])    
        num0_l = int(1/9*len(sourcedata0_left))
        num1_l = int(1/9*len(sourcedata1_left))
        train_data = np.vstack((sourcedata0_left[:2*num0_l], sourcedata1_left[:2*num1_l]))
        validate_data = np.vstack((sourcedata0[2*num0_l:4*num0_l], sourcedata1[2*num1_l:4*num1_l]))
        cand_data = np.vstack((sourcedata0[4*num0_l:], sourcedata1[4*num1_l:]))
    
        train_data = shuffle(train_data)
        cand_data = shuffle(cand_data)
        validate_data = shuffle(validate_data)
        test_data = shuffle(test_data)
            
        
        train_content = train_data[:,0].tolist()
        train_label = train_data[:,1].tolist()
        
        cand_content = cand_data[:,0].tolist()
        cand_label = cand_data[:,1].tolist()
        
        validate_content = validate_data[:,0].tolist()
        validate_label = validate_data[:,1].tolist()
        
        test_content = test_data[:,0].tolist()
        test_label = test_data[:,1].tolist()
        
        
        train_intial = list(zip(train_content, train_label)) 
        cand_intial = list(zip(cand_content, cand_label))
        
        vectorizer = CountVectorizer(stop_words='english')
        
        interactive_output = '../output_sec/iml_al/interact/' + data_name + \
                               '_' + clf_name +'_output_1_'+str(j)+'.csv'        
                            
        csv_file = open(interactive_output, "w", newline='')
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'FPR', 'Precision', 'f_measure',
                     'g_measure', 'mcc', 'accuracy', 'auc', 'PofB20', 'opt'])
        
        iter_num = len(train_label)  
        train_input = train_intial
        cand_input = cand_intial
     
        max_exp = 1
        max_distance = int(thr/100*(len(cand_intial)))
#        print(max_distance)
        perf_value_init = 0
        i = 0
        while 1:
            train_content_input, train_label_input = zip(*train_input)
            train_content_input_list = list(train_content_input)
            train_label_input_list = list(train_label_input)
            
            cand_content_input, cand_label_input = zip(*cand_input)
            cand_content_input_list = list(cand_content_input)
            cand_label_input_list = list(cand_label_input)
            #wxx:standardize the train data and test data,
            train_content_input_matrix = vectorizer.fit_transform(train_content_input_list)
            cand_content_input_matrix = vectorizer.transform(cand_content_input_list)
            validate_content_matrix = vectorizer.transform(validate_content)
            
            #wxx:dimensionality reduction
            train_content_input_matrix_dr, cand_content_input_matrix_dr, validate_content_matrix_dr\
                = dr.selectFromLinearSVC3(train_content_input_matrix,train_label_input, 
                                          cand_content_input_matrix, validate_content_matrix)  
#            #wxx:imbalance process
#            train_content_input_matrix_dr_im, train_label_input_im\
#                = imb.get_uds_nm(train_content_input_matrix_dr,train_label_input)
            
   
            clf.fit(train_content_input_matrix_dr, train_label_input)
            predicted = clf.predict(validate_content_matrix_dr)
            predicted_proba = clf.predict_proba(validate_content_matrix_dr)
            TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, mcc, accuracy, auc, PofB20, opt \
                        = mf.model_measure_with_cross(predicted, predicted_proba, validate_label)    
                
    #        print(iter_num, TP, FN, TN, FP, pd, pf, prec, f_measure,
    #                             g_measure, mcc, accuracy, auc, PofB20, opt)
            writer.writerow([iter_num, TP, FN, TN, FP, pd, pf, prec, f_measure,
                                 g_measure, mcc, accuracy, auc, PofB20, opt])
    
    
            # dynamic stop criteria
            perf_value_now = auc            
            if perf_value_now > perf_value_init:
                perf_value_init = perf_value_now
                i = 0
                train_num_max = iter_num
                train_content_input_max = train_content_input_list
                train_label_input_max = train_label_input
                if perf_value_init >= max_exp:
                    break
            else:
                i += 1
                if i == max_distance or len(cand_label_input) ==0:
                    print('Stop for Stop Criterion reached!')
                    break
            #wxx: standardize and predict the left train data with the model
     
            cand_proba = clf.predict_proba(cand_content_input_matrix_dr)
            cand_proba_list = list()  
            for line in cand_proba: # write predicted left_train data result into left_proba_list
                cand_proba_list.append(line[0])  
            three_column = list(zip(cand_content_input_list, cand_label_input_list, cand_proba_list))
        #     print("============This is the three_column value:\n",three_column)
            #wxx: obtain the unbelievable data according to their performance? value
            min_proba = min(three_column, key=lambda x: abs(x[2]-0.5))   #???
        #     print("============This is the min_proba value:\n",min_proba) 
            #wxx: add unbelievable data into train data(with labeled value)
            train_input.append(min_proba) 
            three_column.remove(min_proba)
            if three_column:
                cand_content_input, cand_label_input, useless_proba = zip(*three_column)
                cand_input = list(zip(cand_content_input, cand_label_input))
            else:
                print('no candidate data left')
                break
            iter_num = iter_num + 1
        csv_file.close()         
        
        train_content_input_matrix_max = vectorizer.fit_transform(train_content_input_max)
        test_content_matrix = vectorizer.transform(test_content)    
        train_content_matrix_max_dr, test_content_matrix_dr\
                = dr.selectFromLinearSVC2(train_content_input_matrix_max,train_label_input_max, test_content_matrix) 
#        train_content_matrix_max_dr_imb, train_label_input_max_imb\
#                = imb.get_uds_nm(train_content_matrix_max_dr,train_label_input_max)
        clf.fit(train_content_matrix_max_dr, train_label_input_max)
        predicted = clf.predict(test_content_matrix_dr)
        predicted_proba = clf.predict_proba(test_content_matrix_dr)    
        TP, FN, TN, FP, pd, pf, prec, f_measure, g_measure, mcc, accuracy, auc, PofB20, opt \
                = mf.model_measure_with_cross(predicted, predicted_proba, test_label)
        writer_o.writerow([train_num_max, TP, FN, TN, FP, pd, pf, prec, f_measure,
                             g_measure, mcc, accuracy, auc, PofB20, opt])  
        print(j,train_num_max,TP, FN, TN, FP, pd, pf, prec, f_measure,
                         g_measure, mcc, accuracy, auc, PofB20, opt)    
    csv_file_k.close()
print("Great! All Finished")
