import pandas
import csv

#data_name = 'camel_perf'

#['ambari','camel','derby','openstack','chromium','wicket']
#['derby','hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec','zookeeper_sec','chromium','openstack']
    
# ['ambari_perf','camel_perf','derby_perf','wicket_perf','hbase_perf','hdfs_perf','cassandra_perf','mapreduce_perf','zookeeper_perf','flume_perf']
#['ambari_perf','camel_perf','derby_perf','wicket_perf']
#==============generate average and max values for results of iml_al, all classifiers =========================
#data_names =['camel','derby','hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec']
#clf_names = ['nb','lr','svm','mlp','rf'] # only for iml_clf 'lr','svm','mlp','rf'
##rate = 20
#output_max = '../output_sec/es/iml_al/results/0_sbr_iml_al_max_auc_result.csv'
##output_max = '../output_sec/es/thr_'+str(rate)+'_iml_al_max_auc_result.csv'
#csv_file_max = open(output_max, "w", newline='')
#writer_max = csv.writer(csv_file_max, delimiter=',')
#writer_max.writerow(['data_name','clf_name','train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 
#                 'precision', 'f_measure', 'auc'])
#
#output_avg = '../output_sec/es/iml_al/results/0_sbr_iml_al_avg_result.csv'
#csv_file_avg = open(output_avg, "w", newline='')
#writer_avg = csv.writer(csv_file_avg, delimiter=',')
#writer_avg.writerow(['data_name','clf_name','train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 
#                 'precision', 'f_measure', 'auc'])
#
#for data_name in data_names:
#    for clf_name in clf_names:
#        datafile = '../output_sec/es/iml_al/results/'+data_name + '_' +clf_name+'.csv'
#        data_result = pandas.read_csv(datafile)[:30]
#        
#        max_AUC = max(data_result.auc)
#        avg_train_num = sum(data_result.train_num)/29
#    #    print(data_result)
#    
#        avg_TP = sum(data_result.TP)/29
#        avg_FN = sum(data_result.FN)/29
#        avg_TN = sum(data_result.TN)/29
#        avg_FP = sum(data_result.FP)/29
#        avg_recall = sum(data_result.Recall) /29
#        avg_precision = sum(data_result.precision) /29
#        avg_f1_score = sum(data_result.f_measure) /29    
#        avg_AUC = sum(data_result.auc) /29
#    
#        row_max_auc = data_result[data_result['auc'] == max_AUC]
#    
#        data = row_max_auc.iloc[[0]].values.tolist()    
##        print(data[0])
#        writer_max.writerow(data[0])    
#        writer_avg.writerow([data_name, clf_name,avg_train_num, avg_TP, avg_FN, avg_TN, avg_FP, avg_recall, avg_precision, avg_f1_score, avg_AUC])
#
#csv_file_max.close()
#csv_file_avg.close()
#print("all datasets done!")
  
#===========================generate results for iml_al, farsec, himpact - from results of 10-fold cross-validation ==============
#['ambari','camel','derby','openstack','chromium','wicket']
#['derby','hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec','zookeeper_sec','chromium','openstack']
   
#data_names = ['ambari_perf','camel_perf','derby_perf','wicket_perf','hbase_perf','hdfs_perf','cassandra_perf','mapreduce_perf','zookeeper_perf']
data_name = 'camel_perf'
#clf_names = ['nb','lr','svm','mlp','rf']
#    rates = [10,20,50]
clf_names = ['lr','svm','mlp','rf']   

for clf_name in clf_names:
    output = '../output_perf/es/iml_al/results/sbr_'+data_name + '_'+clf_name+'_output.csv'
    csv_file_all = open(output, "w", newline='')
    writer_o = csv.writer(csv_file_all, delimiter=',')
    writer_o.writerow(['data_name','clf_name','train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 
                     'precision', 'f_measure', 'auc'])
    
    for m in range(0,30):
    #    print(m)
        datafile = '../output_perf/es/iml_al/' + data_name + \
                               '_' + clf_name +'_output_'+str(m)+'.csv'
        data_result = pandas.read_csv(datafile)
        avg_train_num = sum(data_result.train_num)/10
    #    print(data_result)
    
        avg_TP = sum(data_result.TP)/10
        avg_FN = sum(data_result.FN)/10
        avg_TN = sum(data_result.TN)/10
        avg_FP = sum(data_result.FP)/10
        avg_recall = sum(data_result.Recall) /10
        avg_precision = sum(data_result.Precision) /10
        avg_f1_score = sum(data_result.f_measure) /10    
        avg_AUC = sum(data_result.auc) /10
    
        writer_o.writerow([data_name, clf_name,avg_train_num, avg_TP, avg_FN, avg_TN, avg_FP, avg_recall, avg_precision, avg_f1_score, avg_AUC])
    csv_file_all.close()
print("all datasets done!")    
# =============iml vursus two baselines with average values of 30 times execution=========================
#data_names =['derby','hbase_sec','hdfs_sec','cassandra_sec','mapreduce_sec','zookeeper_sec','chromium','openstack']
#output_avg = '../output_sec/es/iml_vs_bls_result.csv'
#csv_file_avg = open(output_avg, "w", newline='')
#writer = csv.writer(csv_file_avg, delimiter=',')
#writer.writerow(['data_name','clf_name','approach', 'train_num', 'TP', 'FN', 'TN', 'FP', 'Recall', 
#                 'precision', 'f_measure', 'auc'])
#
#for data_name in data_names:
#    datafile_iml = '../output_sec/es/iml_al/latest/0_4es_' + data_name + '_nb.csv'
#    data_result_iml = pandas.read_csv(datafile_iml)
#    
#    avg_train_num_iml = sum(data_result_iml.train_num)/30
##    print(data_result)
#
#    avg_TP_iml = sum(data_result_iml.TP)/30
#    avg_FN_iml = sum(data_result_iml.FN)/30
#    avg_TN_iml = sum(data_result_iml.TN)/30
#    avg_FP_iml = sum(data_result_iml.FP)/30
#    avg_recall_iml = sum(data_result_iml.Recall) /30
#    avg_precision_iml = sum(data_result_iml.precision) /30
#    avg_f1_score_iml = sum(data_result_iml.f_measure) /30    
#    avg_AUC_iml = sum(data_result_iml.auc) /30
#   
#    writer.writerow([data_name, 'nb', 'iBRPredictor', avg_train_num_iml, avg_TP_iml, avg_FN_iml, 
#                     avg_TN_iml, avg_FP_iml, avg_recall_iml, avg_precision_iml, avg_f1_score_iml, avg_AUC_iml])
#
#
#    datafile_farsec = '../output_sec/es/bl_farsec/0_bl_farsec_4es_' + data_name + '.csv'
#    data_result_farsec = pandas.read_csv(datafile_farsec)
#    
#    avg_train_num_fsec = sum(data_result_farsec.train_num)/30
##    print(data_result)
#
#    avg_TP_fsec = sum(data_result_farsec.TP)/30
#    avg_FN_fsec = sum(data_result_farsec.FN)/30
#    avg_TN_fsec = sum(data_result_farsec.TN)/30
#    avg_FP_fsec = sum(data_result_farsec.FP)/30
#    avg_recall_fsec = sum(data_result_farsec.Recall) /30
#    avg_precision_fsec = sum(data_result_farsec.precision) /30
#    avg_f1_score_fsec = sum(data_result_farsec.f_measure) /30    
#    avg_AUC_fsec = sum(data_result_farsec.auc) /30
#   
#    writer.writerow([data_name, 'nb', 'FARSEC', avg_train_num_fsec, avg_TP_fsec, avg_FN_fsec, 
#                     avg_TN_fsec, avg_FP_fsec, avg_recall_fsec, avg_precision_fsec, avg_f1_score_fsec, avg_AUC_fsec])
#
#
#    datafile_hpct = '../output_sec/es/bl_himpact/0_bl_himpact_4es_' + data_name + '.csv'
#    data_result_hpct = pandas.read_csv(datafile_hpct)[:30]
#    print(data_result_farsec)
#    avg_train_num_hpct = sum(data_result_hpct.train_num)/30
##    print(data_result)
#
#    avg_TP_hpct = sum(data_result_hpct.TP)/30
#    avg_FN_hpct = sum(data_result_hpct.FN)/30
#    avg_TN_hpct = sum(data_result_hpct.TN)/30
#    avg_FP_hpct = sum(data_result_hpct.FP)/30
#    avg_recall_hpct = sum(data_result_hpct.Recall) /30
#    avg_precision_hpct = sum(data_result_hpct.precision) /30
#    avg_f1_score_hpct = sum(data_result_hpct.f_measure) /30    
#    avg_AUC_hpct = sum(data_result_hpct.auc) /30
#   
#    writer.writerow([data_name, 'nb', 'High-impact', avg_train_num_hpct, avg_TP_hpct, avg_FN_hpct, 
#                     avg_TN_hpct, avg_FP_hpct, avg_recall_hpct, avg_precision_hpct, avg_f1_score_hpct, avg_AUC_hpct])
#csv_file_avg.close()
#print("all datasets done!")

  


#
# 