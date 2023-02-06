import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
data = pd.read_csv('dataset.csv')

s=data['Weather'] = data['Weather'].replace({'Hot': 0,'Cold':1})


X=data[['Temperature','humidity']]#input varaible

y=data['Weather']#out put varaible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)
   
KNN= KNeighborsClassifier(n_neighbors=3)
predictions=KNN.fit(X_train, y_train)
predictions = KNN.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#New Input Varaible

Temperature=float(input("Enter the Temperature:"))

humidity=float(input("Enter the humidity:"))

testPrediction=KNN.predict([[Temperature,humidity]])

print(testPrediction)


if testPrediction==0:
    print('Hot climate')

else:
    print('Cold Climate')
