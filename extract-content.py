import sys
from langdetect import detect, detect_langs
import pandas as pd


def detect_language(text):
    try:
        # Detect the most probable language
        language = detect(text)
        # Detect probabilities for all possible languages
        probabilities = detect_langs(text)
        return language, probabilities
    except Exception as e:
        return None, str(e)


def detect_lang(row):
    lang, _ = detect_language(row['embedding-content'])
    return lang

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

if len(sys.argv) == 1:
    files = ['csv/small/Business_News-20.csv']
else:
    files = sys.argv[1:]

all_files = []

for item in files:
    dfi = pd.read_csv(item, engine='python')
    all_files.append(dfi)

df = pd.concat(all_files, axis=0, ignore_index=True)

print("CSV file loaded")

# extract the content and tag the language
df['embedding-content'] = df.apply(extract_content, axis=1)
print('content extracted')

df['lang'] = df.apply(detect_lang, axis=1)
print('language tagged')

print(df)

df.to_csv('extracted-content.csv', sep='\t', index=False, encoding='utf-8')
