import pickle
import random
import numpy as np
import tensorflow as tf

# 加载电影特征矩阵
movie_matrics = pickle.load(open('../machine_learning/movie_matrics.p', mode='rb'))
# 加载用户特征矩阵
users_matrics = pickle.load(open('../machine_learning/users_matrics.p', mode='rb'))

# 从本地读取数据
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('../machine_learning/preprocess.p', mode='rb'))
# 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


# 开始预测
def recommend_same_type_movie(movie_id_val, top_k=20):
    norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keepdims=True))
    normalized_movie_matrics = movie_matrics / norm_movie_matrics

    # 推荐同类型的电影
    probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results


def recommend_your_favorite_movie(user_id_val, top_k=10):
    # 推荐您喜欢的电影
    probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])

    print("以下是给您的推荐：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != 5:
        c = np.random.choice(3883, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])

    return results


def recommend_other_favorite_movie(movie_id_val, top_k=20):
    probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
    probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[0][-top_k:]
    #     print(normalized_users_matrics.numpy().shape)
    #     print(probs_user_favorite_similarity.numpy()[0][favorite_user_id])
    #     print(favorite_user_id.shape)

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

    print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
    probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
    probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
    p = np.argmax(sim, 1)
    print("喜欢看这个电影的人还喜欢看：")

    if len(set(p)) < 5:
        results = set(p)
    else:
        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
    for val in (results):
        print(val)
        print(movies_orig[val])
    return results


if __name__ == "__main__":
    # 对于看了电影1401的用户，取得相似度最大的20个做推荐
    print(recommend_same_type_movie(movie_id_val=1401, top_k=20))

    # 看过电影1401的人还喜欢哪些电影
    print(recommend_other_favorite_movie(movie_id_val=1401, top_k=20))

    # 给用户234推荐相似度前十的电影
    print(recommend_your_favorite_movie(user_id_val=234, top_k=10))

"""
def rating_movie(mv_net, user_id_val, movie_id_val):
    categories = np.zeros([1, 18])
    categories[0] = movies.values[movieid2idx[movie_id_val]][2]

    titles = np.zeros([1, sentences_size])
    titles[0] = movies.values[movieid2idx[movie_id_val]][1]

    inference_val = mv_net.model([np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                                  np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                                  np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
                                  categories,
                                  titles])
    return (inference_val.numpy())
    
# 用户234对电影1401的预测评分
print(rating_movie(mv_net, 234, 1401))
"""
