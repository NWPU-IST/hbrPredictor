# -- encoding utf-8 --
 
# 2017-7-27 by xuanyuan14
# 求Kappa系数和Fleiss Kappa系数的模板
# 分集0.0~0.20极低的一致性(slight)、0.21~0.40一般的一致性(fair)、0.41~0.60 中等的一致性(moderate)
# 0.61~0.80 高度的一致性(substantial)和0.81~1几乎完全一致(almost perfect)
 
import numpy as np
def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是kk的
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    Pe  = float(ysum*xsum)/k**2
    P0 = float(P0/k*1.0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))
    return cohens_coefficient
def fleiss_kappa(testData, N, k, n): #testData表示要计算的数据，（N,k）表示矩阵的形状，说明数据是N行j列的，一共有n个标注人员
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0*dataMat[i, j]**2
        temp -= n
        temp /= (n-1)*n
        P0 += temp
    P0 = 1.0*P0/N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i]/sum)**2
    Pe = ysum*oneMat*1.0
    ans = (P0-Pe)/(1-Pe)
    return ans[0, 0]
 
 
##dataArr1 = [[1.1, 1.2], [3.23, 4.78]]
#dataArr2 = [[0, 0, 0, 0, 14],
#                [0, 2, 6, 4, 2],
#                [0, 0, 3, 5, 6]]
input_file1 = "F:\20-source code\0_groudtruth\manual review\"
sourcedata0 = pandas.read_csv(input_file0).fillna('')

dataArr2 = 


#res1 = kappa(dataArr1, 2)
res2 = fleiss_kappa(dataArr2, 10, 5, 3)
#print(res1, res2)
