import sys
from langdetect import detect, detect_langs
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


model = SentenceTransformer('all-MiniLM-L6-v2')


#
# Basic processing
#

def detect_language(text):
    try:
        # Detect the most probable language
        language = detect(text)
        # Detect probabilities for all possible languages
        probabilities = detect_langs(text)
        if False:
            print('--')
            print(text)
            for prob in probabilities:
                print(f"{prob.lang}: {prob.prob:.2f}")
        return language, probabilities
    except Exception as e:
        return None, str(e)

# depends on the global model
def generate_embedding(text):
    embeds = model.encode([text])
    return embeds[0]

def embedPhrases2D(phrases):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(phrases)  # Shape: (n_samples, n_features)
    pca = PCA(n_components=2)
    low_dim_embeddings = pca.fit_transform(embeddings)  # Shape: (n_samples, n_components)
    return low_dim_embeddings


#
# Row processing
#

# NOTE: dependency on schema in here
def filter_en(row):
    lang, _ = detect_language(row['embedding-content'])
    return lang == 'en'


# extract content according to business logic
# assumes we are using a data frame filter with
# a specific schema
def extract_content(row):
    content = row['content']
    desc = row['description']
    title = row['title']
    clen = 0 if pd.isna(content) else len(content)
    dlen = 0 if pd.isna(desc) else len(desc)
    tlen = 0 if pd.isna(title) else len(title)
    if clen > dlen:
        return content
    elif dlen > tlen:
        return desc
    else:
        return title


def embed_content(row):
    content = row['embedding-content']
    embed = generate_embedding(content)
    return embed


# test load
if len(sys.argv) == 1:
    file_path = 'csv/Business_News-20.csv'
else:
    file_path = sys.argv[1]

df = pd.read_csv(file_path)
print("CSV file successfully loaded!")

# extract the content
df['embedding-content'] = df.apply(extract_content, axis=1)
print('content extracted')

# filter for only enlish articles
df = df[df.apply(filter_en, axis=1)]
print('filtered for en')

# now embed
df['embedding'] = df.apply(embed_content, axis=1)
print('content embeded')

print(df)

# reduce to two dimensions
embeddings = df['embedding'].tolist()

pca = PCA(n_components=2)
low_dim_embeddings = pca.fit_transform(embeddings)  # Shape: (n_samples, n_components)
print("Low-Dimensional Embeddings:\n", low_dim_embeddings)
np.savetxt("embeddings-2d.csv", low_dim_embeddings, delimiter=",")

# add the 2d embedding to the df and save, also pull x and y dims separately for ease
df['embedding_2d'] = [row for row in low_dim_embeddings]
df['embedding_x'] = [row[0] for row in low_dim_embeddings]
df['embedding_y'] = [row[1] for row in low_dim_embeddings]
df.to_csv('processed-embeddings.csv', sep='\t', encoding='utf-8')