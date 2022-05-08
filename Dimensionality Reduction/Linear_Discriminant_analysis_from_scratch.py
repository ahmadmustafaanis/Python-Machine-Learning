"""
Step 1 Data Preprocessing of d dimensional data
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
    header=None,
)


X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

"""
Step 2: Computing the scatter matrix
"""
import numpy as np

np.set_printoptions(precision=4)
# looping over class labels(1,2,3) to find mean vector

mean_vec = list()

for label in range(1, 4):
    mean_vec.append(np.mean(X_train_std[y_train == label], axis=0))
    print("MV %s: %s\n" % (label, mean_vec[label - 1]))

# using this mean_vec, we can caluculate within class scatter matrix Sw

d = 13  # number of features
S_W = np.zeros((d, d))
class_scatter = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vec):
    # class_scatter = np.zeros((d,d))
    pass
for row in X_train_std[y_train == label]:
    row, mv = row.reshape(d, 1), mv.reshape(d, 1)

    class_scatter += (row - mv).dot((row - mv).T)

    S_W += class_scatter


# this is not scaled version. We have to Scale the classses

d = 13

S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vec):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter


# Computing Between Class Scatter Matrix SB

mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)
S_B = np.zeros((d, d))


for i, mean_vc in enumerate(mean_vec):

    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vc = mean_vc.reshape(d, 1)

    S_B += n * (mean_vc - mean_overall).dot((mean_vc - mean_overall).T)

eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# sorting the eigen values

eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))
]

eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# measuring how much class linearity is captured by eigenvectors(linear discreminants) via plot

tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

plt.bar(
    range(1, 14), discr, alpha=0.5, align="center", label='induviuvl "discriminability"'
)
plt.step(range(1, 14), cum_discr, where="mid", label='Cumulative "discriminability"')

plt.ylabel('"Discriminability" ratio')
plt.xlabel("Linear Discriminants")
plt.ylim([-0.1, 1.1])
plt.tight_layout()
plt.show()

# since 2 features have all the information, let;s stack them both

w = np.hstack(
    (eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real)
)


X_train_lda = X_train_std.dot(w)  # x' = xw, now it is 2 dimensional

colors = ["r", "b", "g"]

markers = ["s", "x", "o"]


for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_lda[y_train == l, 0],
        X_train_lda[y_train == l, 1] * (-1),
        c=c,
        label=l,
        marker=m,
    )


plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend()

plt.tight_layout()
plt.show()
