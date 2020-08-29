# import pandas as pd
#
# # ratings = pd.read_csv('/Users/a123/Desktop/课程/课设/ml-latest-small/ratings.csv')
# movies = pd.read_csv('./data/movies.csv')
# links = pd.read_csv('./data/links.csv')
# # r_m = ratings.merge(movies, on='movieId')
# #
# # print(ratings[:5])
# # print(movies[:5])
# # print(r_m[:5])
# m = movies.merge(links, on='movieId')
# m = m[:10]
# print(m)
# title = list(m['title'])
# tmdbId = list(m['tmdbId'])
# print(title)
# print(tmdbId)
#
# for t in title:
#     t = t[:-7].replace(' ', '-')
# for i in range(5):
#     print(i)
#
# print(i)

from nltk.corpus import wordnet as wn

print(wn.synset('car.n.01').lemma_names())
# ['car', 'auto', 'automobile', 'machine', 'motorcar']

for synset in wn.synsets('car'):
    print(synset.lemma_names())


