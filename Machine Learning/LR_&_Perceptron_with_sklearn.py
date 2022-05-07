from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris["data"]
y = iris["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

ss = StandardScaler()

ss.fit(X_train)

X_train_std = ss.transform(X_train)
X_test_std = ss.transform(X_test)


class Model:
    def __init__(self, model):
        self.model = model

    def FIT(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test_std)
        print(y_pred)
        print(
            f"Number of Misclassified Examples by {self.model} are {sum(y_pred!=y_test)}"
        )

        print(
            f"Accuracy of {self.model} using Accuracy_score function is {accuracy_score(y_test, y_pred)}"
        )

        print(
            f"Accuracy of {self.model} using Score is {self.model.score(X_test_std, y_test)}"
        )


percp = Perceptron(eta0=0.1, random_state=1)
Logic = LogisticRegression(C=100.0, random_state=1, solver="lbfgs", multi_class="ovr")
Logic2 = LogisticRegression(
    C=100.0,
    random_state=1,
    solver="lbfgs",
    multi_class="multinomial",
)

m1 = Model(percp)
m2 = Model(Logic)
m3 = Model(Logic2)

m1.FIT(X_train_std, y_train)
m2.FIT(X_train_std, y_train)
m3.FIT(X_train_std, y_train)

m1.predict(X_test_std, y_test)
m2.predict(X_test_std, y_test)
m3.predict(X_test_std, y_test)
