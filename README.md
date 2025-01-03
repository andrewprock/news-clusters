stages

1. extract-content.py: extract content and tag language
    -> extracted-content.csv
    adds two columns: embedding-content, lang
2. embed-for-english.py: create embeddings for english news
    -> embedded-content.csv
    adds one column: embedding
3. reduce.py: reduce to low dims
4. cluster.py: cluters in reduced dims
