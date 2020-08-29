from sklearn.datasets import fetch_20newsgroups
from bs4 import BeautifulSoup
from gensim.models import word2vec
import numpy
import nltk
import re
import os

if os.path.exists(r'sentences.npy'):
    # 将预处理后的"词库"从文件中读出，便于调试
    numpy_array = numpy.load('sentences.npy', allow_pickle=True)
    sentences = numpy_array.tolist()
else:
    news = fetch_20newsgroups(subset='all')
    X, y = news.data, news.target
    print("OK")

    # 把段落分解成由句子组成的list（每个句子又被分解成词语）
    def news_to_sentences(news):
        news_text = BeautifulSoup(news, 'lxml').get_text()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(news_text)

        # 对每个句子进行处理，分解成词语
        sentences = []
        for sent in raw_sentences:
            sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
        return sentences


    sentences = []

    for x in X:
        sentences += news_to_sentences(x)

    # 将预处理过的"词库"保存到文件中，便于调试
    numpy_array = numpy.array(sentences)
    numpy.save('sentences.npy', numpy_array)

if os.path.exists(r'word2vec.model'):
    model = word2vec.Word2Vec.load('word2vec.model')
else:
    num_features = 300
    min_word_count = 20
    num_workers = 2
    context = 5
    downsampling = 1e-3

    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)

    model.init_sims(replace=True)  # 模型锁定，可以提高模型和后续任务的速度，同时也使得模型不能训练了，read only

    model.save('word2vec.model')  # 保存模型

# 保存word2vec训练参数便于调试
# model.wv.save_word2vec_format('word2vec_model.bin', binary=True)
# model.wv.load_word2vec_format('word2vec_model.bin', binary=True)

print('词语相似度计算：')
print('morning vs morning:')
print(model.wv.n_similarity('morning', 'morning'))
print('morning vs afternoon:')
print(model.wv.n_similarity('morning', 'afternoon'))
print('morning vs hello:')
print(model.wv.n_similarity('morning', 'hellow'))
print('morning vs shell:')
print(model.wv.n_similarity('morning', 'shell'))

# 以上是用 20newsbydate 和 nltk 的模型，效果很一般
# 以下是用 glove 的转成 word2vec 的模型，效果还行

