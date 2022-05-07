import numpy as np
from decesion_boundary import plot_decision_region
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn import tree


iris = load_iris()
X = iris["data"][:, [2, 3]]
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
# We are not doing featrue scaling but it might be helpful

tree_model = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)

tree_model.fit(X_train, y_train)

X_comb = np.vstack((X_train, X_test))
y_comb = np.hstack((y_train, y_test))
plot_decision_region(X_comb, y_comb, tree_model, test_idx=range(105, 150))

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.tight_layout()
plt.show()


tree.plot_tree(tree_model)
plt.show()


dot_data = export_graphviz(
    tree_model,
    filled=True,
    rounded=True,
    class_names=["Setosa", "versicolor", "Virginica"],
    feature_names=["petal Length", "petal width"],
    out_file=None,
)

graph = graph_from_dot_data(dot_data)
graph.write_png("tree.png")
