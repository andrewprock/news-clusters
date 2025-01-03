import sys
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def embedPhrases2D(phrases):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(phrases)  # Shape: (n_samples, n_features)
    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(embeddings)  # Shape: (n_samples, n_components)
    return low_dim_embeddings


if len(sys.argv) == 1:
    files = ['embedded-content.csv']
    dimensions = 2
else:
    dimensions = int(sys.argv[1])
    files = [sys.argv[2]]

all_files = []
for item in files:
    print(item)
    dfi = pd.read_csv(item, sep='\t', engine='python')
    all_files.append(dfi)

df = pd.concat(all_files, axis=0, ignore_index=True)
# pandas does not like embedding vectors in a csv file, fix
df['embedding'] = df['embedding'].apply(
    lambda x: [float(num) for num in x.strip('[]').split()]
)

#print(df)

print("CSV file loaded")

# reduce to low dimensions
embeddings = df['embedding'].tolist()

#print(embeddings)

pca = PCA(n_components=dimensions)
low_dim_embeddings = pca.fit_transform(embeddings)  # Shape: (n_samples, n_components)
print("Low-Dimensional Embeddings:\n", low_dim_embeddings)
np.savetxt("embeddings-2d.csv", low_dim_embeddings, delimiter=",")

# add the 2d embedding to the df and save, also pull x and y dims separately for ease
df['embedding_2d'] = [row for row in low_dim_embeddings]
df['embedding_x'] = [row[0] for row in low_dim_embeddings]
df['embedding_y'] = [row[1] for row in low_dim_embeddings]
df.to_csv('processed-embeddings.csv', sep='\t', encoding='utf-8')

