import urllib.request
import pandas as pd
import time
import os


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # 注意，这里使用lambda表达式，将文件按照最后修改时间顺序升序排列
        # os.path.getmtime() 函数是获取文件最后修改时间
        # os.path.getctime() 函数是获取文件最后创建时间
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list


movies = pd.read_csv('./data/movies.csv')
links = pd.read_csv('./data/links.csv')
m = movies.merge(links, on='movieId')
# path = './themoviedb'
filelist = get_file_list('./themoviedb')
# print(filelist)

if filelist:  # 目录文件非空，说明已经有些下载的文件
    latestid = filelist.pop()  # 最新下载的文件名
    latestid = int(latestid[:-4])
else:
    latestid = 1
del filelist
line = m[m['movieId'].isin([latestid])].index.values[0]
m = m[line+1:]
# print(m)
movieid = list(m['movieId'])  # 注意此数据集里的movieId是不连续的，所以不能直接用 i 循环
# title = list(m['title'])
tmdbId = list(m['tmdbId'])
count_timeout = 0
for i in range(len(movieid)):  # python3中的range已经是xrange了
    print("\r {} downloading... ".format(movieid[i]), "{} timeout totally".format(count_timeout), end="")
    # tt = title[i]
    # tt = tt[:-7].replace(' ', '-')
    ti = tmdbId[i]
    html = None
    try:
        url = 'https://www.themoviedb.org/movie/' + str(int(ti))  # + '-' + tt
        response = urllib.request.urlopen(url, timeout=5)
        string = response.read()
        html = string.decode('utf-8')
    except:
        try:
            url = 'https://www.themoviedb.org/movie/' + str(int(ti))  # + '-' + tt
            response = urllib.request.urlopen(url, timeout=5)
            string = response.read()
            html = string.decode('utf-8')
        except:
            fd = open("timeout.txt", 'a+')
            fd.write(str(movieid[i])+'\n')
            fd.close()
            count_timeout += 1
    filename = './themoviedb/' + str(movieid[i]) + '.txt'
    if html:
        f = open(filename, 'w')
        f.write(html)
        f.close()
    time.sleep(1)

# UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 46: ordinal not in range(128)
