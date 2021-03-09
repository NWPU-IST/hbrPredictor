# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:22:15 2019

@author: Xiaoxue Wu
"""

import csv
import time

import pandas as pd
import numpy as np

data_name ='Derby'
input_file = "../sourcedata/" + data_name + ".csv"
df = pd.read_csv(input_file).fillna('')

#num=df.loc[0]
summary=df['summary   ']
contents=df['description   ']

print(summary)
#print(contents[109])
##for i in range(0,len(contents)):
##    if len(contents[i]) == 0:
##        print(i)
###        contents[i] = summary[i]
#
#sec = df.Security
#
##print(content)
#output = '../input/' + data_name + '_sec.csv'        
##csv_file = open(output, "w", newline='')
##writer = csv.writer(csv_file, delimiter=',')
##writer.writerow(['ID', 'content','sec'])
#data = pd.DataFrame()
#data['num'] = num
#data['content'] = contents
#data['sec'] = sec
#data.to_csv(output,mode = 'a',index =False)


#csv_file.close()


