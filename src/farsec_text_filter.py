import pandas 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
import operator

'''
# 论文中keywords本身是根据tfidf计算的
keywords = ['org', 'security', 'server', 'test', 'derby', 'permission', 'java', 
   'using', 'access', 'tests', 'error', 'support', 'denied', 'junit', 
   'file', 'exception', 'database', 'code', 'user', 'read', 'manager', 
   'locale', 'run', 'fails', 'need', 'version', 'fail', 'running',   # 将securitymanager改成了security
   'network', 'following', 'files', 'call', 'required', 'statement', 'source', 
   'table', 'connect', 'class', 'thread', 'block', 'exception', 'would', #将securityexception变为security
   'used', 'like', 'failed', 'problem', 'privileged', 'client', 'see', 'jdbc', 'set', 
   'method', 'permission', 'trying', 'granted', 'authentication', 'needs', 'directory', # 将filepermission变为permission
   'connection', 'new', 'think', 'encryption', 'sun', 'jar', 'policy', 'start', 'stack', 'unknown', 
   'rows', 'revoke', 'found', 'information', 'alpha', 'thrown', 'without', 'name', 'one', 'create', 'end', 
   'update', 'http', 'could', 'make', 'trigger', 'though', 'native', 'contains', 'looks', 'two', 'key', 'mode', 
   'results', 'sql', 'use', 'int', 'classpath', 'message', 'incorrectly', 'check']


data_name = 'derby'
#data_name = 'chromium'
train_csv = 'E:/PY/1_dr_research/input/' + data_name + '_train.csv'
test_csv = 'E:/PY/1_dr_research/input/' + data_name + '_test.csv'

col_names = ['content', 'label']
train_data = pandas.read_csv(train_csv, names=col_names, header=None).fillna('')
train_content = train_data.content
train_label = train_data.label

test_data = pandas.read_csv(test_csv, names=col_names, header=None).fillna('')
test_content = test_data.content
test_label = test_data.label
'''
#############################################
def convert(value):
    matched = value.group()
    return matched[0] + ' ' + matched[1]

def Clean_Words(content):

    r = '[a-z][A-Z]'
    content = re.sub(r,convert,content)

    r = '[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    content = re.sub(r,' ',content)
    r = "[0-9]"
    content = re.sub(r,' ',content)
    r = '\s[a-zA-Z0-9]\s'
    content = re.sub(r,' ',content)
    content = ' '.join(content.split())
    
    return content

# 这里的预处理暂不进行词干化
def nltk_words_process_NOstem(content):

    result=[]
    for v in content:
        L = Clean_Words(v).split()
        seq = [s for s in L if s not in stopwords.words('english')]
        s = ' '
        s = s.join(seq)
        result.append(s)
    return result

###############################################



##############################################
# 计算关键词在每一个SBR中的tfidf，然后求和，排序



###########################Score Keywords###############################
def Tf(S,w,Names):
    # 统计w在S中的词频
    index = Names.index(w)
    sum = 0
    for s in S:
        sum = s[index] + sum
    return sum

def Length(S):
    # 统计S中的总词频
    sum = 0
    for i in S:
        for j in i:
            if j > 0:
                sum = sum + j
    return sum
   

def support(a):
    # support本应该是一个处理不平衡问题的函数
    return a

def ScoreWords(B,label,keywords,Names):

    # B是向量化之后的训练集
    # 按照label的值，将训练集分为SBR和NSBR
    # 论文中的partition部分
    S = []
    NS = []
    for i,j in enumerate(label):
        if j == 1:
            S.append(B[i])
        else:
            NS.append(B[i])

    Score = { }
    for w in keywords:
        p_sw = min(1,support(Tf(S,w,Names) / Length(S)))
        p_nsw = min(1,support(Tf(NS,w,Names) / Length(NS)))
        Score[w] = max(0.01,min(0.99,p_sw / (p_sw + p_nsw)))

#    print('关键词评分')
#    print(sorted(Score.items(),key=operator.itemgetter(1),reverse=True))

    return Score


# print(ScoreWords(train_content_matrix,support))
#########################Score Bug Report##############################

# 阶乘
# 更正为求和
def Factorial(M):
    r = 0
    for m in M:
        # r = r * m
        r = r + m
    return r


def ScoreReport(R,M):
    # R是一个bug report
    
    # 本来的数据是没有做大小写处理的
    R = R.lower().split()
    # 对于R中的每一个word，如果在keywords字典中返回字典对应的值，否则返回0
    # 但是这里如果返回0的话，会导致后面的计算也是0
    # 因此将阶乘的函数从阶乘变成了求和
    M1 = [M.get(w,0) for w in R]
    M2 = [1 - m for m in M1]
    if (Factorial(M1) + Factorial(M2)) >0:
        return Factorial(M1) / (Factorial(M1) + Factorial(M2))
    else:
        return 0


# 获取bug分数，返回的是分数较高且label为0的bug的位置索引
def Get_Report_Score(content,label,M,s):
    L = []
    for i,r in enumerate(content):
#        print("************************index is: ", i, "content is : \n", r)
        L.append((ScoreReport(r,M),label[i],i))
    L = sorted(L,key=operator.itemgetter(0),reverse=True)
    r = []
#    print('评分，label，在原训练集中的位置==========', len(L))
    for i in L:
#        print(i)
        # 这里的阈值s可以根据经验值调整
        if i[0] > s and i[1] == 0:
            r.append(i[2])
#    print("===============",len(r),"::::::::::::", r)
    return r

# 对训练集和对应的label进行删除处理
# 训练集文本，训练集标签，关键词表的词个数，计算bug report分数后的阈值
def FARSEC(train_content,train_label,num,val):
#    print(len(train_content))

    content = nltk_words_process_NOstem(train_content)
    label=train_label
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    content_matrix = vectorizer.fit_transform(content)
    tfidf = transformer.fit_transform(content_matrix).toarray()

    tfidf_score = {}
    Names = vectorizer.get_feature_names()
#    print("names have got: \n", Names)
    for n in Names:
        tfidf_score[n] = 0  # 初始化tfidf值
    for i,l in enumerate(label):
        if l == 0:
            continue
        for j,n in enumerate(Names):
            tfidf_score[n] = tfidf_score[n] + tfidf[i][j]

    tfidf_score = sorted(tfidf_score.items(),key=operator.itemgetter(1),reverse=True)

    # 关键字的数量也可以进行调整
    keywords = [i[0] for i in tfidf_score[:num]]
#    print("==========keywords:\n",keywords)

    content_matrix = content_matrix.toarray()
    M = ScoreWords(content_matrix,label,keywords,Names)
#    print("=======================number of M:  ",len(M), "=============\n", M)
    s = Get_Report_Score(content,label,M,val)

    a_index = [i for i in range(0,len(train_content))]
    a_index = set(a_index)
    b_index = set(s)
    index = list(a_index-b_index)
    print("the list index: ", index)
    train_content = [train_content[i] for i in index]
    train_label= [train_label[i] for i in index]
#    print("******************",len(train_content))

#    train_content.drop(index=s, inplace=True)
#    train_label.drop(index=s, inplace=True)
#    train_content = train_content.reset_index().remove(columns='index').content# 重置索引,将原索引删除
#    train_label = train_label.reset_index().remove(columns='index').label
    return train_content,train_label
