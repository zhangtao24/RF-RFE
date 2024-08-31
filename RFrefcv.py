# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score


csvdata = pd.read_csv(r"D:\studysoft\python_project\paper2_RFE\100.csv")

featureName = []
for i in csvdata.columns:
    featureName.append(i)

data = np.array(csvdata)

#分割训练集合测试集  X第二列往后所有数据  y第一列所有数据
X = data[:,1:]
y = data[:,0]

#随机森林评估特征重要性
feat_labels=featureName[1:]

#构造随机森林模型
forest= RandomForestClassifier(n_estimators=100,random_state=0)

#选出最好的随机森林模型
#rfc_list=[]       #分类器各参数
#superpa = []      #结果值
#for i in range(1,201):
#   rfc = RandomForestClassifier(n_estimators=i)                #每次交叉验证所得值，评价指标为准确率，这里取平均
#   rfc_s = cross_val_score(rfc,X,y,cv=5,scoring='accuracy')      #保存每次交叉验证的平均准确率
#   superpa.append(rfc_s.mean())
#   rfc_list.append(rfc)
#print('最大交叉验证平均值：',max(superpa),'\n第',superpa.index(max(superpa))+1,'个随机森林评估器')
#best_rfc=rfc_list[superpa.index(max(superpa))]
#print('最好的评估器:',best_rfc)
#plt.figure(figsize=[10,6])
##最大范围range+1
#plt.plot(range(1,201),superpa)
#plt.show()
#print(superpa)
# #最好的评估器模型
# forest=best_rfc
#拟合模型
forest.fit(X,y)
#
#括号中放入测试集特征，可以输出对应的预测结果
#print('预测值：',forest.predict(X))
#
#feature_importances_:特征重要性
importances=forest.feature_importances_
#np.argsort：从小到大排序，输出排序对应的索引
indices=np.argsort(importances)[::-1]
#
for f in range(X.shape[1]):
    #给予10000颗决策树平均不纯度衰减的计算来评估特征重要性·1
    print ("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))
#
#
# Create the RFE object and compute a cross-validated score.
rfecv = RFECV(estimator=forest, step=1, cv=StratifiedKFold(5),scoring='accuracy')
rfecv.fit(X, y)
# #
#rfecv.n_features_：具有交叉验证的选定特征的数量
print("Optimal number of features（最佳特征数） : %d" % rfecv.n_features_)
#rfecv.ranking_：特征排名
print("Ranking of features（特征排名） : %s" % rfecv.ranking_)
a = rfecv.ranking_
b = rfecv.grid_scores_
print('rfecv.grid_scores_:',rfecv.grid_scores_)
#
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, marker='o')
plt.show()
#
result = pd.DataFrame([feat_labels,importances,a,b])
#
print(result)
#保存结果
result.to_csv(r"D:\studysoft\python_project\paper2_RFE\10011.csv")