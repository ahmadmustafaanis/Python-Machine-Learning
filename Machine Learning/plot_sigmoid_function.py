import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color="k")
plt.ylim(-0.1, 1.1)
plt.xlabel("z")
plt.ylabel("$\phi (z)$")
plt.yticks([0, 0.5, 1])

ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
