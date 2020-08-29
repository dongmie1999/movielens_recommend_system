# from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# from gensim.scripts.glove2word2vec import glove2word2vec


def most_similar(word, topn):
    # 加载转化后的文件
    if r'glove2word2vev.model':
        model = KeyedVectors.load('glove2word2vev.model')
    # else:
    #     try:
    #         model = KeyedVectors.load_word2vec_format('/Users/a123/Desktop/glove.6B/word2vec_300d.txt')
    #     except FileNotFoundError:
    #         # 输入文件
    #         glove_file = datapath('/Users/a123/Desktop/glove.6B/glove.6B.300d.txt')
    #         # 输出文件
    #         tmp_file = get_tmpfile("word2vec_300d.txt")
    #         #
    #         # call glove2word2vec script
    #         # default (through CLI):python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
    #         #
    #         # 开始转换
    #         print(glove2word2vec(glove_file, tmp_file))
    #
    #         # 在glove下载的txt这个文件的最开头，加上两个数
    #         # 第一个数指明一共有多少个向量，第二个数指明每个向量有多少维，就能直接用word2vec的load函数加载了
    #         model = KeyedVectors.load_word2vec_format(tmp_file)

    try:
        r = model.most_similar(word, topn=topn)
        model.most_similar_to_given()
    except KeyError:  # 未收录这个词
        r = None
    return r

if __name__ == "__main__":
    most_similar()
