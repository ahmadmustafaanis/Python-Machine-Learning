from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from plot_decesion_region import plot_decision_regions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


'''
Data Pre processing
'''

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)


X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=0)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

colors = ['r','b','g']

markers=['s', 'x','o']


for l,c,m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_lda[y_train==l, 0],
                X_train_lda[y_train==l, 1]* (-1),
                c=c, label=l, marker=m)


plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend()

plt.tight_layout()
plt.show()

lr = LogisticRegression(multi_class='ovr', random_state=1, solver = 'lbfgs')


lr.fit(X_train_lda[0:2,:], y_train[0:2])

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.title("Before Fitting Model")
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

lr.fit(X_train_lda, y_train)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.title("After Fitting Model")
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
