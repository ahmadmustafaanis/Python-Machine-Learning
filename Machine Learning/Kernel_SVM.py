"""
We will use a dataset that is in the form of XOR gate using logical_or function from numpy
"""

# generating the data set
from decesion_boundary import plot_decision_region
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


np.random.seed(1)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c="b", marker="x", label="1")

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c="r", marker="s", label="-1")

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend()
plt.tight_layout()
plt.show()

# just use 'rbf' kernel instead of linear kernel in sklearn to implement Gaussian Kernel is SVM
gamma = float(input("Type value of Gamma for SVM\n"))
C = int(input("Type Value of C for SVM\n"))

<<<<<<< HEAD:Kernel_SVM.py
svm = SVC(C=C, kernel="rbf", gamma=gamma, random_state=1)

svm.fit(X_xor, y_xor)
=======
svm.fit(X_xor,y_xor)
>>>>>>> 1932f91746c6ca348280af77c18e94e76606e7f9:Machine Learning/Kernel_SVM.py
plot_decision_region(X_xor, y_xor, classifier=svm)
plt.tight_layout()
plt.show()
