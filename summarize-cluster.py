import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.metrics.pairwise import cosine_similarity
import pdb

# set up nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# load the labelled clusters
df = pd.read_csv('labelled-embeddings.csv', sep='\t', engine='python')

labels = df['labels'].unique()
labels.sort()

pd.set_option('display.max_colwidth', 100)
print(labels)

label_scores = {}

for label in labels:
    print('----', label, '----')
    cluster = df.loc[df['labels'] == label]
    cluster_content = cluster['embedding_content']
    print(cluster['title'])


    # decode embeddings
    dfe = cluster['embedding'].apply(
        lambda x: [float(num) for num in x.strip('[]').split(',')]
    )
    num_columns = len(dfe.iloc[0])
    column_names = range(num_columns)
    dfc = pd.DataFrame(dfe.to_list(), columns=column_names)

    # now compute centroid and similarities
    centroid = np.mean(dfc.to_numpy(), axis=0).reshape(1, -1)
    cum = 0
    count = 0
    for index, row in dfc.iterrows():
        rowvals = dfc.iloc[index].values.reshape(1, -1)
        similarity = cosine_similarity(rowvals, centroid)
        cum += similarity
        count += 1
    cum /= count
    label_scores[label] = cum

    # normalize text
    as_text = ' '.join(cluster_content.astype(str).values.flatten())
    words = word_tokenize(as_text)
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()
    cleaned_words = [
        stemmer.stem(word.lower())  # Apply stemming to the word
        for word in words
        if len(word) >= 6 and word.lower() not in stop_words and word not in punctuation
    ]

    # get the top tokens
    fdist = FreqDist(cleaned_words)
    top_words = fdist.most_common(8)
    print(top_words)

for key, value in label_scores.items():
    print(key, value)

min_item = min(label_scores.items(), key=lambda x: x[1])
print(min_item)