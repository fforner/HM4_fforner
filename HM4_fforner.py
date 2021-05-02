import numpy as np
import pandas as pd  
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
#print(boston.keys())
#print(boston.DESCR)
#print(boston.target)

X = pd.DataFrame(boston.data, columns = boston.feature_names)
Y = pd.DataFrame(boston.target, )
Y.columns= ['MEDV']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2)

Reg=LinearRegression()
Reg.fit(X_train, Y_train)

R_2_test = Reg.score(X_test,Y_test)
R_2_train = Reg.score(X_train,Y_train)
print(f'This is the regression accuracy for the train model: {R_2_train}')
print(f'This is the regraasion accuracy for the test model: {R_2_test}')

estimates = Reg.coef_
print(f'Those are the regression parametrs: {estimates}')


############ part 2 ############
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, load_iris
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

wine = load_wine()
iris = load_iris()

X_wine = pd.DataFrame(wine.data, columns = wine.feature_names)
Y_wine = pd.DataFrame(wine.target)

X_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
Y_iris = pd.DataFrame(iris.target)


#wine data set
sse = {}
for nk in range(1,10):
    total = 0.0
    kmeans = KMeans(n_clusters= nk, max_iter=1000).fit(X_wine)
    X_wine["clusters"] = kmeans.labels_
    center = kmeans.cluster_centers_
    sse[nk] = kmeans.inertia_

plt.figure()
plt.style.use("fivethirtyeight")
plt.plot(list(sse.keys()), list(sse.values()))
plt.title('Wine Elbow Method Graph')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


print(f'The elbow point is the point where the SSE curve begins to bend.\nThe x-value of this point is thought to be a good compromise between error and cluster count.\nThe elbow is located at x = 3 in this case.')


# iris dataset
sse2 = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_iris)
    X_iris["clusters"] = kmeans.labels_
    center = kmeans.cluster_centers_
    sse2[k] = kmeans.inertia_

# Plot to see where the elbow is to pick optimal number of populations
plt.figure()
plt.style.use("fivethirtyeight")
plt.plot(list(sse2.keys()), list(sse2.values()))
plt.title('Iris Elbow Method Graph')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

print("number of clusters on the data is 4")



