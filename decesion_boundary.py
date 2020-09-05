import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt 

def plot_decision_region(X_train, X_test,y_train, y_test, classifier, test_idx = None, resolution = 0.02):
    X = np.vstack((X_train, X_test))
    y = np.hstack((y_train, y_test))
    
    #setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plotting the decision boundary

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1

    xx1 ,xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
               np.arange(x2_min, x2_max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx1.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1],
        alpha=0.8, c=colors[idx],
        marker=markers[idx], label=cl,
        edgecolor='black')
    #highlight test examples

    if test_idx:
        # plot all examples

        X_test, y_test = X[test_idx,:], y[test_idx]

        plt.scatter(X_test[:,0], X_test[:,1],
        c='', edgecolor='black', alpha=1.0, 
        linewidth=1, marker='o', 
        s=100, label='test_set'
        )