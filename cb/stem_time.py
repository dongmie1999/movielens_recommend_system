from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import pandas as pd
import time

tag = pd.read_csv('./data/tags.csv')
tag = list(tag['tag'])
start = time.time()
for t in tag:
    stemmer = PorterStemmer()
    stemmer.stem(t)
print(time.time()-start)

start = time.time()
for t in tag:
    stemmer2 = SnowballStemmer("english")
    stemmer2.stem(t)
print(time.time()-start)
