from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
#Load dataset
iris=datasets.load_iris()
print("Iris Data set loaded...")
# print(iris)
# Split the data into train and test samples
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.1)
print("Dataset is split into training and testing...")
print("size of trainng data and its label",x_train.shape,y_train.shape)
print("Size of trainng data and its label",x_test.shape, y_test.shape)
# Prints Label no. and their names
for i in range(len(iris.target_names)):
      print("Label", i, "-",str(iris.target_names[i]))
      # Create object of KNN classifier
classifier = KNeighborsClassifier(n_neighbors=1)
#Perform Training
classifier.fit(x_train, y_train)
# Perform testing
y_pred=classifier.predict(x_test)
# Display the results
print("Results of Classification using K-nn with K=1")
for r in range(0,len(x_test)):
      print(" Sample:", str(x_test[r]), " Actual-label:", str(y_test[r]), "Predicted-label:", str(y_pred[r]))
      print("Classification Accuracy:", classifier.score(x_test,y_test));

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,y_pred))
print('Accuracy Metrics')
print(classification_report(y_test,y_pred))



from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.1,random_state=15)

k=int(input('Enter k: '))

pred=[]

for i in x_test:
  d=[]
  for j in x_train:
    s=np.sqrt(np.sum((i-j)**2))
    d.append(s)
  indices=np.argsort(d)
  y=y_train[indices]
  c={}
  for q in range(k):
    c[y[q]]=c.get(y[q],0)+1
  x=-1
  z=-1
  for key,value in c.items():
    if value>x:
      x=value
      z=key
  pred.append(z)



print('Predicted labels: ',pred)
print('Actual class labels: ',list(y_test))

correct=0
for i in range(len(pred)):
  if pred[i]==y_test[i]:
    correct+=1

acc=(correct/len(pred))*100
print('Accuracy = %.3f'%(acc))
