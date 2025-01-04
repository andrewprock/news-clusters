stages

1. extract-content.py: extract content and tag language
    -> extracted-content.csv
    adds two columns: embedding_content, lang
2. embed-for-english.py: create embeddings for english news
    -> embedded-content.csv
    adds one column: embedding
3. reduce.py: reduce to low dims
    -> embeddings-2d.csv
    though the actual dims may not be 2
4. cluster.py: cluters in reduced dims
5. python summarize-cluster.py: produces a summary report