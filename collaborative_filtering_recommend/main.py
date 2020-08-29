import pandas as pd
import os
from math import sqrt
import time

if os.path.isfile("../content_based_recommend/data/data.csv"):
    pass
else:
    movies = pd.read_csv('../content_based_recommend/data/movies.csv')
    ratings = pd.read_csv('../content_based_recommend/data/ratings.csv')  # 这里注意如果路径的中文件名开头是r，要转义。
    data = pd.merge(movies, ratings, on='movieId')  # 通过两数据框之间的movieId连接
    data[['userId', 'rating', 'movieId', 'title']].sort_values('userId').to_csv('../content_based_recommend/data/data.csv', index=False)

file = open("../content_based_recommend/data/data.csv", 'r', encoding='UTF-8')  # 记得读取文件时加‘r’， encoding='UTF-8'
# 读取data.csv中每行中除了名字的数据
data = {}  ##存放每位用户评论的电影和评分
for line in file.readlines():
    # 注意这里不是readline()
    line = line.strip().split(',')
    # 如果字典中没有某位用户，则使用用户ID来创建这位用户
    if not line[0] in data.keys():
        data[line[0]] = {line[3]: line[1]}
    # 否则直接添加以该用户ID为key字典中
    else:
        data[line[0]][line[3]] = line[1]
# print(data)
data.pop('userId')
"""
data是读取的csv文件，得的很快不存为.npy也可以
这里的得到的data是二维字典的形式，类似以下形式
{..., '2':{'Inception (2010)': '4.0',
           'Shutter Island (2010)': '4.0'...}, ...}
"""

"""
计算任何两位用户之间的相似度，由于每位用户评论的电影不完全一样，所以兽先要找到两位用户共同评论过的电影
然后计算两者之间的欧式距离，最后算出两者之间的相似度
"""


def Euclidean(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    # 找到两位用户都评论过的电影，并计算欧式距离
    for key in user1_data.keys():
        if key in user2_data.keys():
            # 注意，distance越小表示两者越相似
            distance += pow(float(user1_data[key]) - float(user2_data[key]), 2)

    return 1 / (1 + sqrt(distance))  # 这里返回值越小，相似度越大


# 计算某个用户与其他用户的相似度
def top5_similiar(userID):
    res = []
    for userid in data.keys():
        # 排除与自己计算相似度
        if not userid == userID:
            simliar = Euclidean(userID, userid)
            res.append((userid, simliar))
    res.sort(key=lambda val: val[1])
    return res[:5]


"""
top5_simliar的返回的列表形式如，按相似度升序排序
[('68',  0.044330050969940915),
 ('599', 0.04807925798778345),
 ('217', 0.04843346156984026),
 ('160', 0.050181926468153115),
 ('474', 0.050965218942982136)]
"""

# print(len(data))
# print(top5_similiar('1'))
# print(top5_similiar('13'))


# 计算两用户之间的Pearson相关系数
def pearson_sim(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    common = {}

    # 找到两位用户都评论过的电影
    for key in user1_data.keys():
        if key in user2_data.keys():
            common[key] = 1
    if len(common) == 0:
        return 0  # 如果没有共同评论过的电影，则返回0
    n = len(common)  # 共同电影数目
    # print(n, common)

    # 计算评分和
    sum1 = sum([float(user1_data[movie]) for movie in common])
    sum2 = sum([float(user2_data[movie]) for movie in common])

    # 计算评分平方和
    sum1Sq = sum([pow(float(user1_data[movie]), 2) for movie in common])
    sum2Sq = sum([pow(float(user2_data[movie]), 2) for movie in common])

    # 计算乘积和
    PSum = sum([float(user1_data[it]) * float(user2_data[it]) for it in common])

    # 计算相关系数
    num = PSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r

# R = pearson_sim('1', '3')
# print(R)


def recommend(uid, amount=5):
    uid = str(uid)
    s_people = top5_similiar(uid)  # 相似用户，形如（uid:相似度）[('68', 0.044),('599', 0.048)]
    # print(s_people)
    u_seen = data[uid].keys()  # 用户看过的电影的名称
    recommend = {}  # 初始化推荐字典
    for person in s_people:
        s_seen = data[person[0]].keys()  # 相似用户看过的电影名称
        for sseen in s_seen:  # 遍历相似用户看过的电影
            if sseen not in u_seen:  # 如果当前电影目标用户没看过，推荐
                try:
                    recommend[sseen] += person[1] * float(data[person[0]][sseen])
                except KeyError:
                    # print(data[person[0]])
                    # print(sseen)
                    # print(person)
                    recommend[sseen] = person[1] * float(data[person[0]][sseen])
    # recommend.sort(key=lambda kv:(kv[1], kv[0]))
    return sorted(recommend.items(), key=lambda kv: (kv[1], kv[0]))[-amount:]


if __name__ == '__main__':
    start = time.time()
    for i in range(602):
        try:
            print(recommend(i+1))
        except:
            pass
    print("time used:", time.time()-start)
