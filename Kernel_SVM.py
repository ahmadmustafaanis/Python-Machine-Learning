'''
We will use a dataset that is in the form of XOR gate using logical_or function from numpy
'''

# generating the data set
from decesion_boundary import plot_decision_region
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0] > 0, X_xor[:,1 ]>0)

y_xor = np.where(y_xor,1 , -1)

plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], c='b', marker='x', label='1')

plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],c='r', marker='s', label='-1')

plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend()
plt.tight_layout()
plt.show()

#just use 'rbf' kernel instead of linear kernel in sklearn to implement Gaussian Kernel is SVM

from sklearn.svm import SVC 
svm = SVC(C=10.0, kernel='rbf',gamma=0.10, random_state=1)

svm.fit(X_xor, y_xor)

plot_decision_region(X_xor, y_xor, classifier=svm)
plt.tight_layout()
plt.show()