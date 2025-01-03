import sys
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


model = SentenceTransformer('all-MiniLM-L6-v2')

# depends on the global model
def generate_embedding(text):
    embeds = model.encode([text])
    return embeds[0]


#
# Row processing
#

# NOTE: dependency on schema in here
def filter_en(row):
    return row['lang'] == 'en'


def embed_content(row):
    content = row['embedding-content']
    embed = generate_embedding(content)
    return embed


if len(sys.argv) == 1:
    files = ['extracted-content.csv']
else:
    files = sys.argv[1:]

all_files = []
for item in files:
    dfi = pd.read_csv(item, sep='\t', engine='python')
    all_files.append(dfi)

df = pd.concat(all_files, axis=0, ignore_index=True)

print("CSV file loaded")

# filter for only enlish articles
df = df[df.apply(filter_en, axis=1)]
print('filtered for en')

# now embed
df['embedding'] = df.apply(embed_content, axis=1)
print('content embeded')

df.to_csv('embedded-content.csv', encoding='utf-8')
