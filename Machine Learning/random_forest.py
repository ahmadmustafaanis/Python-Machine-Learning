from sklearn.ensemble import  RandomForestClassifier
import numpy as np
from decesion_boundary import plot_decision_region
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = load_iris()
X = iris['data'][:, [2,3]]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1, stratify =y)


forest = RandomForestClassifier(criterion='gini', n_estimators=10, random_state=1, n_jobs=2)

forest.fit(X_train, y_train)

X_comb = np.vstack((X_train, X_test))
y_comb = np.hstack((y_train, y_test))
plot_decision_region(X_comb, y_comb, forest, test_idx=range(105,150))

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.tight_layout()
plt.show()