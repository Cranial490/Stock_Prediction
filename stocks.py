import numpy as np
from numpy import genfromtxt
from sklearn import svm 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


dataset = genfromtxt('finalData.csv',delimiter=',')
feat = dataset[:,:6]
output = dataset[:,7,None]
#feat,output = shuffle(feat,output,random_state=102)
standard_scaler = StandardScaler()
feat = standard_scaler.fit_transform(feat)
X_train = feat[450:,:]
Y_train = output[450:,:]
X_test = feat[1:450,:]
Y_test = output[1:450,:]
holdout = feat[451:700,:]
holdoutY = output[451:700,:]
clf = svm.SVC(kernel = 'poly',degree=9)
clf.fit(X_train,Y_train)
predicted =  clf.predict(X_test)

print accuracy_score(Y_test,predicted)