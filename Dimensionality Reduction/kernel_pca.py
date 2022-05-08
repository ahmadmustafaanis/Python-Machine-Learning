import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from plot_decesion_region import plot_decision_regions

X, y = make_moons(n_samples=100)

print(X.shape, y.shape)
print()

print(
    "Number of Unique Classes", len(np.unique(y)), "\nUnique Values are", np.unique(y)
)

kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)

X_kpca = kpca.fit_transform(X)
plt.style.use("ggplot")
plt.scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color="red", marker="^", alpha=0.5)

plt.scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color="blue", marker="o", alpha=0.5)


plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()


lr = LogisticRegression()

lr.fit(X_kpca, y)

plot_decision_regions(X_kpca, y, lr)
plt.xlabel("Axis 1")
plt.ylabel("Axis 2")
plt.tight_layout()
plt.show()
plt.show()
