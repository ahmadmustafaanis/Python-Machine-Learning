'''
Importing important libraries
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from plot_decesion_region import plot_decision_regions
import matplotlib.pyplot as plt

# initializing PCA transformer and LogisticRegression

pca = PCA(n_components=2)

LR = LogisticRegression(multi_class='ovr', random_state=1, solver='lbgfs')

'''
Step 1: Loading and Pre-Processing the Data
'''

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=0)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# fitting the logisitic regression model on the databases

lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')

lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.tight_layout()

plt.show()

print(lr.score(X_test_pca, y_test))
