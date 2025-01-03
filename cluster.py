import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
from pandas import DataFrame
from sklearn import datasets
from sklearn.mixture import GaussianMixture


clist = list(matplotlib.colors.CSS4_COLORS.keys())
random.shuffle(clist)
print(clist)


# turn it into a dataframe
df = pd.read_csv('embeddings-2d.csv', header=None)


cluster_count = 50

gmm = GaussianMixture(n_components = cluster_count)

# Fit the GMM model for the dataset
# which expresses the dataset as a
# mixture of 3 Gaussian Distribution
gmm.fit(df)

# Assign a label to each sample
labels = gmm.predict(df)
df['labels']= labels

labels = []
for cluster in range(0,cluster_count):
    dx = df[df['labels']== cluster]
    labels.append(dx)
    plt.scatter(dx[0], dx[1], c=clist[cluster])

# for label in labels:
#     plt.scatter(label[0], label[1])


plt.show()