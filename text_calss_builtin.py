import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
msg=pd.read_csv('naivetext.csv',names=['message','label']) #Tabular form dta
print('Total instances in the dataset:',msg.shape[0])

msg['labelnum']=msg.label.map({'pos':1,'neg':0})
X=msg.message
Y=msg.label

print('\nThe message and its label of first 5 instances are listed below')
X5, Y5 = X[0:5], msg.label[0:5]
for x, y in zip(X5,Y5):
    print(x,',',y)
    
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)
print('\nDataset is split into Training and Testing samples')
print('Total training instances :', xtrain.shape[0])
print('Total testing instances :', xtest.shape[0])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain) #Sparse matrix
xtest_dtm = count_vect.transform(xtest)
print('\nTotal features extracted using CountVectorizer:',xtrain_dtm.shape[1])

xtrain_dtm

print('\nFeatures for first 5 training instances are listed below')
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df[0:5])

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
print('\nClassstification results of testing samples are given below')

for doc, p in zip(xtest, predicted):
    pred = 'pos' if p==1 else 'neg'
    print('%s -> %s ' % (doc, pred))
    
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix
print('\nAccuracy metrics')
print('Accuracy of the classifer is',accuracy_score(ytest,predicted))
print('Recall:{0}\n Precison:{1}'.format(recall_score(ytest,predicted, pos_label='positive',average='micro'),precision_score(ytest,predicted,pos_label='positive',average='micro')))
print('Confusion matrix')
print(confusion_matrix(ytest,predicted))