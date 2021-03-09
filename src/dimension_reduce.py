'''
Created on 2018年10月24日

@author: Administrator
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

def get_pca_standard(train_content, test_content):
    pca = PCA(n_components =200)
    new_train = pca.fit_transform(train_content.todense())
    new_test = pca.transform(test_content.todense())
    return new_train, new_test

def selectFromLinearSVC(data, label):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(data, label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    new_data = model.transform(data)
    return new_data
def selectFromLinearSVC3(train_content, train_label, content_1, content_2):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1',max_iter=10000).fit(train_content,train_label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)    
    new_train = model.transform(train_content)
    new_content_1 = model.transform(content_1)
    new_content_2 = model.transform(content_2)
    return new_train, new_content_1,new_content_2

def selectFromLinearSVC4(train_content, train_label, cand_content, validate_content, test_content):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(train_content,train_label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    
    new_train = model.transform(train_content)
    new_cand = model.transform(cand_content)
    new_validate = model.transform(validate_content)
    new_test = model.transform(test_content)
    return new_train, new_cand,new_validate,new_test

def selectFromLinearSVC2(train_content, train_label, test_content):
    lsvc = LinearSVC(C=0.3, dual=False, penalty='l1').fit(train_content,train_label)  # , dual=False  plants 
    model = SelectFromModel(lsvc, prefit=True)
    
    new_train = model.transform(train_content)
    new_test = model.transform(test_content)
 
    return new_train, new_test

