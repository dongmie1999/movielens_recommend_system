import pandas as pd
# import feature_cast_crew
from gensim.models import KeyedVectors
import numpy as np
import os.path


# # 获取演职员名单索引和压缩特征向量
# cast_crew_index, cast_crew_compressed_matrix = feature_cast_crew.get_feature_vector('./themoviedb/')
# print(cast_crew_index)
# print(cast_crew_compressed_matrix)


def get_feature_genres():
    if os.path.isfile('../content_based_recommend/genres_index.npy'):
        genres_index = np.load('../content_based_recommend/genres_index.npy', allow_pickle=True).tolist()
        genres_compressed_matrix = np.load('../content_based_recommend/genres_compressed_matrix.npy', allow_pickle=True).item()
    else:
        # 读取csv文件
        movies_csv = pd.read_csv('../content_based_recommend/data/movies.csv')
        # 初始化电影种类以及其movieid
        movieid = list(movies_csv['movieId'])
        genres = list(movies_csv['genres'])

        # 创建电影种类索引
        genres_index = []
        for genre in genres:
            split = genre.split('|')
            for s in split:
                if s not in genres_index:
                    genres_index.append(s)
        # print("genres_index")
        # print(genres_index)

        # 创建电影压缩特征矩阵
        genres_compressed_matrix = {}
        for i in range(len(movieid)):
            movie_compressed_matrix = []
            genre = genres[i]
            split = genre.split('|')
            for s in split:
                position = genres_index.index(s)
                movie_compressed_matrix.append(position)
            genres_compressed_matrix[movieid[i]] = movie_compressed_matrix
        # print("genres_compressed_matrix")
        # print(genres_compressed_matrix)

        # 保存为csv和npy文件方便阅读和读取
        pd.DataFrame(data=genres_index).to_csv('genres_index.csv', encoding='gbk')
        pd.DataFrame(data=genres_compressed_matrix).to_csv('genres_compressed_matrix.csv', encoding='gbk')
        np.save('../content_based_recommend/genres_index.npy', np.array(genres_index))
        np.save('../content_based_recommend/genres_compressed_matrix.npy', genres_compressed_matrix)
    # return genres_index, genres_compressed_matrix
    return genres_compressed_matrix


def get_feature_tags():
    if os.path.isfile('../content_based_recommend/tags_index.npy'):
        # tags_index = np.load('../content_based_recommend/tags_index.npy', allow_pickle=True).tolist()
        tags_compressed_matrix = np.load('../content_based_recommend/tags_compressed_matrix.npy', allow_pickle=True).item()
    else:
        tags_csv = pd.read_csv('../content_based_recommend/data/tags.csv')
        # 初始化电影tags以及其movieid
        tagged_movieid = list(tags_csv['movieId'])
        tags = list(tags_csv['tag'])

        # 创建标签索引
        tags_index = []
        for tag in tags:
            l = []
            if ' ' not in tag:  # 如果只有一个单词
                tag = tag.lower()
            l.append(tag)
            if l not in tags_index:
                tags_index.append(l)
        # print("tags_index")
        # print(tags_index)

        # 近义词合并
        model = KeyedVectors.load('../content_based_recommend/glove2word2vev.model')
        # print("Start looking for synonyms")
        i = 0
        while i < len(tags_index):  # 这里的tags_index长度是变化的，不能用for，第二重循环同
            print("\r {} in the processing... ".format(i), end="")
            current_tag = tags_index[i]  # 当前要比对是否有近义词的词
            j = i + 1
            while j < len(tags_index):
                pending_tag = tags_index[j]  # pending_tags是list. 从当前词的后一位开始，逐一比对看是不是近义词
                try:
                    topn = model.most_similar(current_tag[0])
                    for similar in topn:
                        if pending_tag[0] == similar[0]:  # 不要后面的那个相似度
                            current_tag.append(pending_tag[0])
                            del tags_index[j]  # 删除了列表元素，长度变化
                except KeyError:
                    break
                j += 1
            i += 1
        # print("tags_index")
        # print(tags_index)

        # 保存
        pd.DataFrame(data=tags_index).to_csv('../content_based_recommend/tags_index.csv', encoding='gbk')
        np.save('../content_based_recommend/tags_index.npy', np.array(tags_index))

        tags_compressed_matrix = {}
        for j in range(len(tagged_movieid)):
            movie_compressed_matrix = []
            tag = tags[j]
            # position = tags_index.index(tag)
            position = None
            for k in range(len(tags_index)):
                tag_and_similar = tags_index[k]
                if tag in tag_and_similar or tag.lower() in tag_and_similar or tag.capitalize() in tag_and_similar:
                    position = k
                    break
            try:  # 如果这个movieid已经创建了一部分向量
                tags_compressed_matrix[tagged_movieid[j]].append(position)
            except KeyError:  # 这个movieid还没创建向量
                tags_compressed_matrix[tagged_movieid[j]] = []
                tags_compressed_matrix[tagged_movieid[j]].append(position)
        # print("tags_compressed_matrix")
        # print(tags_compressed_matrix)

        # 保存为csv和npy文件方便阅读和读取
        pd.DataFrame(data=tags_compressed_matrix).to_csv('../content_based_recommend/tags_compressed_matrix.csv', encoding='gbk')
        np.save('../content_based_recommend/tags_compressed_matrix.npy', tags_compressed_matrix)
    # return tags_index, tags_compressed_matrix
    return tags_compressed_matrix


if __name__ == '__main__':
    get_feature_tags()
