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
#import imbalance_strategies as imb
import model_measure_functions as mf
from scipy import sparse

    
def get_classifier(clf_name):
    if clf_name == 'lr':
        clf = LogisticRegression()
    elif clf_name == 'svm':
        # the kernel can also be 'linear', 'rbf','polynomial','sigmoid', etc.
        clf = svm.SVC(kernel='linear', probability = True)
    #        clf = svm.SVC(kernel='rbf', probability = True, class_weight = 1, decision_function_shaope = 'ovr')
    elif clf_name == 'mlp':
        clf = MLPClassifier(max_iter=5000,shuffle = True)
    elif clf_name == 'nb':
        clf = MultinomialNB()
    elif clf_name == 'rf':
        clf = RandomForestClassifier(oob_score=True, n_estimators=30)
    else:
        print('分类器名称仅为\'lr,svm,mlp,nb,rf\'中的一种')
    return clf


rate = 50 # can be 10, 20, 50,  and 100 only for 'nb'

data_name ='CWE119'
clf_name = 'rf' #,'nb','lr','svm','mlp','rf'

print(data_name, " Begin ",  clf_name)
input_file = "../input-vul/" + data_name + ".csv"
sourcedata = pandas.read_csv(input_file).fillna('')
sourcedata = shuffle(sourcedata)

content = sourcedata.code
label = sourcedata.label.tolist()
#print(content)
#content = sourcedata.Description

#print(len(content), len(label), num0)
clf = get_classifier(clf_name)    


kfold_output = '../output-vul/' + data_name + \
                       '_rf_output.csv'        
              
csv_file_k = open(kfold_output, "w", newline='')
writer_o = csv.writer(csv_file_k, delimiter=',')
writer_o.writerow(['train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 'Precision', 'f1_measure','auc'])



test_content = content[:500]
test_label = label[:500]

validate_content =  content[500:1000]
validate_label = label[500:1000]

train_content =  content[1000:1200]
train_label = label[1000:1200]
 
cand_content =  content[1200:]
cand_label = label[1200:]        

#print(test_label)

train_intial = list(zip(train_content, train_label)) 
cand_intial = list(zip(cand_content, cand_label))

vectorizer = CountVectorizer(stop_words='english')

iter_num = len(train_label)  
train_input = train_intial
cand_input = cand_intial
 
max_exp = 1
rate_tmp = rate/100
max_distance = int(rate_tmp*(len(cand_intial)))
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
    
    #wxx:train a model 
    train_content_input_matrix_dr, cand_content_input_matrix_dr, validate_content_matrix_dr\
        = dr.selectFromLinearSVC3(train_content_input_matrix,train_label_input, cand_content_input_matrix, validate_content_matrix)  
   
    clf.fit(train_content_input_matrix_dr, train_label_input)
#        print(train_content_matrix_input_dmr.shape)
    predicted = clf.predict(validate_content_matrix_dr)

    TP, FN, TN, FP, pd, prec, f_measure, auc \
                = mf.model_measure_f1_auc(predicted, validate_label)    
    print('**********pd is: ',pd,f_measure)
    # dynamic stop criteria
    perf_value_now = pd            
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

train_content_input_matrix_max = vectorizer.fit_transform(train_content_input_max)
test_content_matrix = vectorizer.transform(test_content)    
train_content_matrix_max_dr, test_content_matrix_dr\
        = dr.selectFromLinearSVC2(train_content_input_matrix_max,train_label_input_max, test_content_matrix)  
clf.fit(train_content_matrix_max_dr, train_label_input_max)
predicted = clf.predict(test_content_matrix_dr)
#        predicted_proba = clf.predict_proba(test_content_matrix_dr)    
TP, FN, TN, FP, pd, prec, f_measure, auc \
        = mf.model_measure_f1_auc(predicted, test_label)
writer_o.writerow([train_num_max, TP, FN, TN, FP, pd, prec, f_measure, auc])  
print(train_num_max,TP, FN, TN, FP, pd, prec, f_measure,auc)    
csv_file_k.close()
print("Great! All Finished")
