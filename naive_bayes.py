import numpy as np
import pandas as pd
t=pd.read_csv('play_tennis.csv')
from sklearn.model_selection import train_test_split
x=t.drop('Class',axis=1)
y=t.Class
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=15)
c=len(x_train)
print('No. of rows in x_train =',c)
q=len(x_test)
print('No. of rows in x_test =',q)
k=len(x_train.columns)

x_train.index=range(c)
x_test.index=range(q)
y_train.index=range(c)
y_test.index=range(q)
print('Train data')
print(x_train)
print('\nTest data')
print(x_test)

c_n=len(y_train[y_train=='N'])
p_n=c_n/c
print('Probability of Negative class = ',p_n)
c_p=len(y_train[y_train=='P'])
p_p=c_p/c
print('Probability of Positive class = ',p_p)

d_p={}
d_n={}
x_train['Class']=y_train
col=x_train.columns
for i in range(k):
  l=x_train[col[i]].unique()
  for j in l:
    d_p[j]=len(x_train[(x_train[col[i]]==j) & (x_train.Class=='P')])/c_p
    d_n[j]=len(x_train[(x_train[col[i]]==j) & (x_train.Class=='N')])/c_n
res=[]
p,n=1,1
for i in range(q):
  f=x_test.loc[i]
  for j in range(k):
    p=p*d_p[f[j]]
    n=n*d_n[f[j]]
  p=p*p_p
  n=n*p_n
  if p>=n:
    res.append('P')
  else:
    res.append('N')
    
print('Predicted class labels =',res)
print('Actual class labels =\n',y_test)
acc=0
for i in range(q):
  if res[i]==y_test[i]:
    acc=acc+1

print('Accuracy = %.3f'%(acc/q))