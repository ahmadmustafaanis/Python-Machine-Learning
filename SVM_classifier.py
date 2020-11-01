import matplotlib.pyplot as plt
from sklearn.svm import SVC
from decesion_boundary import plot_decision_region
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris['data'][:, [2,3]]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1, stratify =y)

ss = StandardScaler()

ss.fit(X_train)

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)


svm = SVC(C=1.0, kernel='linear', random_state=1, verbose=1)

svm.fit(X_train_std, y_train)

print(svm.score())