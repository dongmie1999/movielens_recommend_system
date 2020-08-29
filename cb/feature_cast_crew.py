import re
import os
import numpy as np
import pandas as pd


def get_file_list(file_path):
    """
    :param file_path: the file path where you want to get file
    :return: list, files sorted by name
    """
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        # dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        dir_list = sorted(dir_list, key=lambda x: int(x[:-4]))  # 按名称排序
        # print(dir_list)
        return dir_list


def txt_match(filename):
    """
    :param filename: filename of the html file that contain the cast and crew
    :return: list, the names of cast and crew, like ['John Lasseter', 'Alec Sokolow', 'Andrew Stanton', 'Joss Whedon']
    """
    f = open(filename, "r")  # 打开test.txt文件，以只读的方式
    data = f.readlines()  # 循环文本中得每一行，得到得是一个列表的格式<class 'list'>
    f.close()  # 关闭test.txt文件
    crew = []
    for line in data:
        if re.match(".*?Crew</a></p>", line):
            break
        result = re.match(".*?>.*?</a></p>", line)
        if result:
            rule = r'">(.*?)</a></p>'
            slotlist = re.findall(rule, result.string)
            if slotlist:
                crew.append(slotlist[0])
    try:
        p = crew.pop()  # 检查最后一项是否异常，每个列表值只检查一次
        if not re.match("View More.*?</span>", p):  # 有时候列表最后一项是'View More...</span>'
            crew.append(p)
    except IndexError:  # 有些网站上没有演职员名单
        pass
    return crew


def get_feature_vector(filepath='../content_based_recommend/themoviedb/'):
    """
    analyze the html file and return compressed feature vector.
    :param filepath: str, the folder in which html files are stored
    :return:
    cast_crew_index: list, index of cast & crew, like [['John Lasseter', 'Alec Sokolow', 'Andrew Stanton']
    cast_crew_compressed_matrix: dict, the position of non-zero elements in the list
    """
    if os.path.isfile('../content_based_recommend/cast_crew_compressed_matrix.npy'):
        cast_crew_index = np.load('../content_based_recommend/../content_based_recommend/cast_crew_index.npy', allow_pickle=True).tolist()
        cast_crew_compressed_matrix = np.load('../content_based_recommend/cast_crew_compressed_matrix.npy', allow_pickle=True).item()
    else:
        # 创建电影特征向量索引
        cast_crew_index = []
        filelist = get_file_list(filepath)  # 按修改时间排序的文件
        # print(filelist)
        for filename in filelist:
            # 选出每部电影的演职员名单
            cast_crew = txt_match(filepath + filename)  # my filepath is './themoviedb/'
            # print(filename, cast_crew)
            if cast_crew:
                for c in cast_crew:
                    if c not in cast_crew_index:  # 还没记录这个人，记录下来
                        cast_crew_index.append(c)
        # print(cast_crew_index)
        # 保存
        pd.DataFrame(data=cast_crew_index).to_csv('../content_based_recommend/cast_crew_index')  # 后面的encoding不写默认为utf-8
        np.save('../content_based_recommend/cast_crew_index.npy', np.array(cast_crew_index))

        # 创建电影压缩矩阵
        cast_crew_compressed_matrix = {}
        for i in range(len(filelist)):
            movieid = int(filelist[i][:-4])
            movie_cast_crew = txt_match('../content_based_recommend/themoviedb/' + filelist[i])
            movie_compressed_matrix = []  # 初始化当前电影的压缩特征向量
            # 查找每部电影的演职员名单在总的电影特征向量中的位置并保存
            for c in movie_cast_crew:
                position = cast_crew_index.index(c)  # 查找演员在特征向量索引里的位置
                movie_compressed_matrix.append(position)  # 保存
            # movie_dict = {movieid: movie_compressed_matrix}
            # cast_crew_compressed_matrix.update(movieid=movie_compressed_matrix)
            cast_crew_compressed_matrix[movieid] = movie_compressed_matrix

        # 保存
        # pd.DataFrame(data=cast_crew_compressed_matrix).to_csv('cast_crew_compressed_matrix', encoding='gbk')
        np.save('../content_based_recommend/cast_crew_compressed_matrix.npy', np.array(cast_crew_compressed_matrix))

    # print(cast_crew_compressed_matrix)
    # return cast_crew_index, cast_crew_compressed_matrix
    return cast_crew_compressed_matrix


get_feature_vector()

# def fab(max):
#     n, a, b = 0, 0, 1
#     while n < max:
#         yield b  # 使用 yield
#         # print b
#         a, b = b, a + b
#         n = n + 1
#
#
# for n in fab(5):
#     print(n)