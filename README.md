# hbrPredictor

hbrPredictor, a tool used for high impact bug report prediction with uncertainty-sampling.

=============================================================

Wu, Xiaoxue, Wei Zheng, Xiang Chen, Yu Zhao, Tingting Yu, and Dejun Mu. "Improving high-impact bug report prediction with combination of interactive machine learning and active learning." Information and Software Technology 133 (2021): 106530.

 - Context: Bug reports record issues found during software development and maintenance. A high-impact bug report (HBR) describes an issue that can cause severe damage once occurred after deployment. Identifying HBRs from the bug repository as early as possible is crucial for guaranteeing software quality. 
 - Objective: In recent years, many machine learning-based approaches have been proposed for HBR prediction, and most of them are based on supervised machine learning. However, the assumption of supervised machine learning is that it needs a large number of labeled data, which is often difficult to gather in practice. 
 - Method: In this paper, we propose hbrPredictor, which combines interactive machine learning and active learning to HBR prediction. On the one hand, it can dramatically reduce the number of bug reports required for prediction model training; on the other hand, it improves the diversity and generalization ability of training samples via uncertainty sampling. Result: We take security bug report (SBR) prediction as an example of HBR prediction and perform a large-scale experimental evaluation on datasets from different open-source projects. The results show: 
   * (1) hbrPredictor substantially outperforms the two baselines and obtains the maximum values of F1-score (0.7939) and AUC (0.8789); 
   * (2) with the dynamic stop criteria, hbrPredictor could reach its best performance with only 45% and 13% of the total bug reports for small-sized datasets and large-sized datasets, respectively. 
 
 - Conclusion: By reducing the number of required training samples, hbrPredictor could substantially save the data labeling effort without decreasing the effectiveness of the model.
