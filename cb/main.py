import pandas as pd
from content_based_recommend import feature_cast_crew
from content_based_recommend import feature_genres_and_tags
# import numpy as np
import time

class content_based_recommend:
    """
    content based recommendation systerm.
    uid: userid, should be included in dataset
    recommend_num: int, how many movies will be returned
    """
    def __init__(self, uid, recommend_num=3):
        self.uid = uid
        self.recommend_num = recommend_num
        self.ulist = []

    def get_user_topn(self, topn=10):
        """
        :param topn: int, how many movies will be returned
        :return: list, movieid of user's favorite, default 3
        """
        # uid = None
        # while type(uid) != int:
        #     try:
        #         uid = int(input("please input an integer:"))
        #     except ValueError:
        #         uid = input("please input an integer:")

        ratings_csv = pd.read_csv('../content_based_recommend/data/ratings.csv')
        start = 0
        udict = {}
        for index in ratings_csv.index:
            if self.uid == int(ratings_csv.loc[index].values[0]):
                start = index
                break
        for index in ratings_csv.index:
            if self.uid == int(ratings_csv.loc[index + start].values[0]):
                udict[int(ratings_csv.loc[index + start].values[1])] = int(ratings_csv.loc[index + start].values[2])
            else:
                break
        self.ulist = sorted(udict.items(), key=lambda x: x[1])
        topnlist = []
        for i in range(topn):
            topnlist.append(self.ulist[-i-1][0])
        return topnlist

    def get_score(self, ucm, dcm):
        score = 0
        for u in ucm:
            if u in dcm:
                score += ucm.count(u) + dcm.count(u)
        return score

    def main(self):
        topnlist = self.get_user_topn()
        # print(topnlist)
        databasedict = {}
        # 对用户最喜欢的电影，在数据库中进行逐一比对
        for topn in topnlist:
            # # 获取索引和压缩特征向量，为计算方便（尽量少占用内存）暂不用索引
            # cast_crew_index, cast_crew_compressed_matrix = feature_cast_crew.get_feature_vector()
            cast_crew_compressed_matrix = feature_cast_crew.get_feature_vector()
            genres_compressed_matrix = feature_genres_and_tags.get_feature_genres()
            tags_compressed_matrix = feature_genres_and_tags.get_feature_genres()
            c, g, t = [], [], []
            try:
                c = cast_crew_compressed_matrix[topn]
            except KeyError:
                pass
            try:
                g = genres_compressed_matrix[topn]
            except KeyError:
                pass
            try:
                t = genres_compressed_matrix[topn]
            except KeyError:
                pass
            for key in cast_crew_compressed_matrix.keys():
                if key not in self.ulist:  # 目标用户没看过这部电影
                    cast_crew_score, genres_score, tags_score = 0, 0, 0
                    try:  # 可能会有些电影没被打过标签，在字典里索引不到
                        cast_crew_score = self.get_score(c, cast_crew_compressed_matrix[key])  # key就是这里的，肯定有
                        genres_score = self.get_score(g, genres_compressed_matrix[key])  # 电影种类是数据集给定的，也都有
                        tags_score = self.get_score(t, tags_compressed_matrix[topn])  # 标签可能没人打过，可能没有
                    except KeyError:
                        pass
                    databasedict[key] = cast_crew_score + genres_score + tags_score

        return sorted(databasedict.items(), key=lambda x: x[1])[-self.recommend_num:]


if __name__ == '__main__':
    # start = time.time()
    # for uid in range(10):
    #     print(content_based_recommend(uid+1).main())  # 打印uid的推荐列表，按照推荐匹配度升序排列
    # print("time used:", time.time()-start)
    print(content_based_recommend(0).main())
