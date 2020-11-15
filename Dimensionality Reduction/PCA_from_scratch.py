import pandas as pd


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)


#step 1 is standardizing the data
''' Train Test Split, Standard Scaler'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=0)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#step 2,3
''' This Step is to create a covariance matrix, and extract the eigen values'''
# we have to construct a symmetric dxd matrix where d is number of columns(dimensions) in the orignal dataset
#this matrix stores the pairwise covariance between the features
#we will caluculate eigen values from this
import numpy as np
cov_matrix = np.cov(X_train_std.T)
#print("Shape of COv matrix is ", cov_matrix.shape)
eigen_vals, eigen_vectors = np.linalg.eig(cov_matrix)

print("Eigen Values", eigen_vals,sep='\n')

'''
Explaining the eigen vectors varianve ratio via plots
'''
import matplotlib.pyplot as plt

tot = sum(eigen_vals)
exp_var = [i/tot for i in sorted(eigen_vals, reverse=True)]

cumulative_var_exp = np.cumsum(exp_var)

plt.bar(range(1,14), exp_var, alpha=0.5, align='center', label='Induvivual variance Explained')

plt.step(range(1,14), cumulative_var_exp, where='mid', label='Cumulative Explained Variance')

plt.ylabel('Explained Variance Ration')
plt.xlabel("Principal Component index")
plt.legend()
plt.tight_layout()
plt.show()

'''
Step 4: Sorting the Eigenpairs(eigen val, eigen vec) in descending order by eigen values
Then Construct a projection matrix from selected eigen vecotrs, and use the projection to transform the data onto the lower dimensions subspace
'''

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))]

eigen_pairs.sort(key=lambda k: k[0], reverse=True)

'''since first 2 eigen vectors have 60% of variance, we will use them'''
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

#print("Matrix", w)


#using this projecttion, we can transform an example x(13 dimenisonal) to onto pca, and getting x` which is 2 dimensional
#x` = xW
# or
#print(X_train_std[0].dot(w))

'''
Similarly we can transform the entire 124x13 dimensional training dataset onto Principal Component
'''

X_train_pca = X_train_std.dot(w)

#print("Entire Training Set after PCA ", X_train_pca, sep='\n')
print("Training Set Has Shape", X_train_pca.shape)

''' Let's Visualize'''

colors=['r','b','g']
markers=['s','x','o']


for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca2.png', dpi=300)
plt.show()
