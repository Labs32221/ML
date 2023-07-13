import numpy as np
import pandas as pd
data=pd.read_csv('naivetext.csv')
data['Text']=data['Text'].str.lower()
from sklearn.model_selection import train_test_split
x=data
y=data.Class
t,test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=15)
r=len(t)
c=len(t.columns)
print('No. of rows in train set :',r)

r1=len(test)
test.index=range(r1)
print('No. of rows in test set:',r1)

t.index,test.index,y_train.index,y_test.index=range(r),range(r1),range(r),range(r1)
vocab=[]
for i in range(r):
  vocab.extend(t['Text'][i].split())

vocab=list(set(vocab))
l=len(vocab)
print('Vocabulary:',vocab)
c_p=len(t[t.Class=='pos'])
c_n=len(t[t.Class=='neg'])
p_p,p_n=c_p/r,c_n/r
print('Probability of positive texts = %.3f'%(p_p))
print('Probability of negative texts = %.3f'%(p_n))


d_p,d_n={},{}
for i in range(r):
  if t['Class'][i]=='pos':
    for j in t['Text'][i].split():
      d_p[j]=d_p.get(j,0)+1
  else:
    for j in t['Text'][i].split():
      d_n[j]=d_n.get(j,0)+1

n1=sum(d_p.values())
n2=sum(d_n.values())
for i in vocab:
  d_p[i]=(d_p.get(i,0)+1)/(n1+l)
  d_n[i]=(d_n.get(i,0)+1)/(n2+l)


res=[]
for i in range(r1):
  p,n=1,1
  for j in test['Text'][i].split():
    if j in vocab:
      p=p*d_p[j]
      n=n*d_n[j]

  p=p*p_p
  n=n*p_n
  if p>n:
    res.append('pos')
  else:
    res.append('neg')
    
    tn,fn,tp,fp=0,0,0,0
for i in range(r1):
  if res[i]=='neg' and test['Class'][i]=='neg':
    tn=tn+1
  elif res[i]=='neg' and test['Class'][i]=='pos':
    fn=fn+1
  elif res[i]=='pos' and test['Class'][i]=='neg':
    fp=fp+1
  elif res[i]=='pos' and test['Class'][i]=='pos':
    tp=tp+1

cfm=[[tn,fp],[fn,tp]]
print('Confusion Matrix:',cfm)
acc=(tp+tn)/(tp+tn+fp+fn)
precision=tp/(tp+fp)
recall=tp/(tp+fn)

print('Accuracy = %.3f'%(acc))
print('Precision = %.3f'%(precision))
print('Recall = %.3f'%(recall))
