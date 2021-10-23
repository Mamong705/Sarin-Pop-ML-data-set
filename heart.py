#เรียกใช้ library
import pandas as pd  
import numpy as np  
from sklearn.linear_model import LogisticRegression as Lori
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

#Prepare Data
ir = pd.read_csv('/content/heart_failure_clinical_records_dataset.csv',usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12])
X = ir.drop(columns=['DEATH_EVENT'])[:300]
z = ir['DEATH_EVENT'][:300]

model = [Lori(max_iter=1000,solver='lbfgs'),DecisionTreeClassifier(max_leaf_nodes=3, random_state=2, criterion='entropy')]
withi = [u'Logistic Regression',u'DecisionTreeClassifier']

#K-fold
Score = [[],[]]
splits = 25
for f,t in StratifiedKFold(n_splits=splits).split(X,z):
  for i,m in enumerate(model):
    Score[i].append(m.fit(X.iloc[f],z.iloc[f]).score(X.iloc[t],z.iloc[t])*100)
Score_Mean = np.mean(Score,1)

print('')
print('ทดสอบ Accuracy ของเทคนิค k-fold , Segment = ' + str(splits))
for i in range(2):
  print('%s: %.1f%%'%(withi[i],Score_Mean[i]))
print('')

# เเบ่งข้อมูลออกเป็นชุดเรียนรู้เเละชุดทดสอบโดยกำหนดให้ชุดทดสอบมีขนาดเป็น 20% ของข้อมูล
x_train,x_test,y_train,y_test = train_test_split(X,z,test_size=0.20,random_state=2)

#LogisticRegression
LoriModel = Lori(max_iter=1000,solver='lbfgs') #prepareModel
LoriModel.fit(x_train,y_train) #TrainModel
y_pre = LoriModel.predict(x_test)
print('ทดสอบ Precision และ Recall ของเทคนิค Evaluation metrics โดยใช้ LogisticRegression')
print("Precision Score : ",metrics.precision_score(y_test,y_pre,average='micro')*100,'%')
print("Recall Score : ",metrics.recall_score(y_test,y_pre,average='micro')*100,'%')
print("Accuracy : ",metrics.accuracy_score(y_test,y_pre)*100,'%')
print('')

#DecisionTreeClassifier
DTC=DecisionTreeClassifier(max_leaf_nodes=3, random_state=2, criterion='entropy') #prepareModel
DTC.fit(x_train,y_train) #TrainModel
k_pred=DTC.predict(x_test)
print('ทดสอบ Precision และ Recall ของเทคนิค Evaluation metrics โดยใช้ DecisionTreeClassifier')
print("Precision Score : ",metrics.precision_score(y_test,k_pred,average='micro')*100,'%')
print("Recall Score : ",metrics.recall_score(y_test,k_pred,average='micro')*100,'%')
print("Accuracy : ",metrics.accuracy_score(y_test,k_pred)*100,'%')
print('')

# Input feature
while True:
  print('age :')
  age = float(input())
  print('anaemia :')
  anaemia = float(input())
  print('creatinine_phosphokinase :')
  creatinine_phosphokinase = float(input())
  print('diabetes :')
  diabetes = float(input())
  print('ejection_fraction :')
  ejection_fraction = float(input())
  print('high_blood_pressure :')
  high_blood_pressure = float(input())
  print('platelets:')
  platelets = float(input())
  print('serum_creatinine :')
  serum_creatinine = float(input())
  print('serum_sodium :')
  serum_sodium = float(input())
  print('sex :')
  sex = float(input())
  print('smoking :')
  smoking = float(input())
  print('time :')
  time = float(input())

  x_test2 = [[age,anaemia,creatinine_phosphokinase,diabetes ,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time]]
  LoriPredict = LoriModel.predict(x_test2)
  DreTreee=DTC.predict(x_test2)

  print('The results of the predictions (Logistic Regression) : ' + str(LoriPredict))
  print('The results of the predictions (DecisionTree Classifier) : ' + str(DreTreee))
  print()
