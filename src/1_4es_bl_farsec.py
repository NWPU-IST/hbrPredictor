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

import imbalance_strategies as imb
import model_measure_functions as mf
import farsec_text_filter as farsec

    
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

data_name ='cassandra'
clf_name = 'nb' #,'nb','lr','svm','mlp','rf'

print(data_name, " Begin ",  clf_name)
input_file = "../input-1/" + data_name + ".csv"
sourcedata = pandas.read_csv(input_file).fillna('')
#sourcedata = sourcedata.sort_values(by="bugid", ascending=False)
#(key=lambda a: a["key"])
content = sourcedata.Description
label = sourcedata.config.tolist()

num0 = int(1/6*len(label))
print(len(content), len(label), num0)
clf = get_classifier(clf_name)    

for m in range(0,3):
    kfold_output = '../output-1/bl-farsec/config/' + data_name + \
                           '_' + clf_name +'_output_rus_'+str(m)+'.csv'        
                        
    csv_file_k = open(kfold_output, "w", newline='')
    writer_o = csv.writer(csv_file_k, delimiter=',')
    writer_o.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'Precision', 'F_measure', 'AUC'])
    for j in range(0,5):
#        print('**********this is j: ',j)
        train_content = content[j*num0:(j+1)*num0]      
        train_label = label[j*num0:(j+1)*num0]
#        print(len(train_content), len(train_label))
        train_content_fs,train_label_fs = farsec.FARSEC(train_content, train_label, 50, 0.1)
#        print(len(train_content_fs))
        if j<4:
#            print("j < 4")
            test_content = content[(j+1)*num0:(j+2)*num0]  
            test_label = label[(j+1)*num0:(j+2)*num0] 
        else:
#            print("j==4")
            test_content = content[(j+1)*num0:]  
            test_label = label[(j+1)*num0:] 
#            print(test_content)
            
        vectorizer = CountVectorizer(stop_words='english') 
        train_content_matrix = vectorizer.fit_transform(train_content_fs)        
        test_content_matrix = vectorizer.transform(test_content)   
#        data_content_matrix_smt, data_label_smt = imb.get_ovs_smote_standard(train_content_matrix, train_label)   
#        data_content_matrix_smt, data_label_smt = imb.get_uds_rdm(train_content_matrix, train_label)         
        clf.fit(train_content_matrix, train_label_fs)
        predicted = clf.predict(test_content_matrix)  
#        print(predicted,test_label)
        TP, FN, TN, FP, pd, prec, f_measure, auc \
                = mf.model_measure_f1_auc(predicted, test_label)
        writer_o.writerow([len(train_label),TP, FN, TN, FP, pd, prec, f_measure, auc])  
        print(len(train_label),TP, FN, TN, FP, pd, prec, f_measure,auc)   
    print(m)
    csv_file_k.close()
print("All finished for dataset: ", data_name)

