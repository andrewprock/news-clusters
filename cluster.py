import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn.mixture import GaussianMixture
import sys

random.seed(1)


clist = list(matplotlib.colors.CSS4_COLORS.keys())
random.shuffle(clist)

# extract the low dim vectors from the dsv
dfx = pd.read_csv('processed-embeddings.csv', sep='\t', engine='python')
dfx['embedding_lowd'] = dfx['embedding_lowd'].apply(
    lambda x: [float(num) for num in x.strip('[]').split()]
)
num_columns = len(dfx['embedding_lowd'].iloc[0])
column_names = range(num_columns)
df = pd.DataFrame(dfx['embedding_lowd'].to_list(), columns=column_names)

# Fit a GMM model to the dataset
cluster_count = 10
if len(sys.argv) > 1:
    cluster_count = int(sys.argv[1])

gmm = GaussianMixture(n_components = cluster_count)
gmm.fit(df)

# extract labels and save
labels = gmm.predict(df)
df['labels'] = labels

if True:
    # plot the clusters
    labelz = []
    for cluster in range(0,cluster_count):
        dx = df[df['labels']== cluster]
        labelz.append(dx)
        plt.scatter(dx[0], dx[1], c=clist[cluster%len(clist)])
    for label in labelz:
        plt.scatter(label[0], label[1])
    plt.show()

dfx['labels'] = labels

dfx.to_csv('labelled-embeddings.csv', index=False, sep='\t', encoding='utf-8')
