from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')
data=[[7,7],[5,9],[6,10],[3,8],[4,10],[5,7]]
label=[['bad'],['bad'],['bad'],['good'],['good'],['good']]
fn=KNeighborsClassifier(n_neighbors=3)
fn.fit(data,label)
a=int(input("enter the data"))
b=int(input("enter the data"))
predicted=fn.predict([[a,b]])
print("predicted",predicted)

