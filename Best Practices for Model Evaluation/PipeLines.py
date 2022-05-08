import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
    header=None,
)

X = df.loc[:, 2:]
y = df.loc[:, 1]

le = LabelEncoder()
y = le.fit_transform(y)

# dividing the dataset


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1, test_size=0.2
)

print(X_train.shape)

# Standardizing, then making the pipeline


# pipeline takes in arbitrary number of Sklearn Transformers(having fit and transform function, followed by a estimator(fit and predict) function)
pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(solver="lbfgs", random_state=1),
    verbose=True,
)

pipe_lr.fit(X_train, y_train)

y_pred = pipe_lr.predict(X_test)

print("Test Accuracy: ", f"{pipe_lr.score(X_test, y_test):.2f}")
