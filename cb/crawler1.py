import urllib.request
import time

f = open('timeout.txt', 'r')
count_timeout = 0
for timeoutid in f.readlines():
    timeoutid = timeoutid.strip('\n')  # 去掉列表中每一个元素的换行符
    print("\r {} downloading... ".format(timeoutid), "{} timeout totally".format(count_timeout), end="")
    html = None
    try:
        url = 'https://www.themoviedb.org/movie/' + str(int(timeoutid))  # + '-' + tt
        response = urllib.request.urlopen(url, timeout=5)
        string = response.read()
        html = string.decode('utf-8')
    except:
        try:
            url = 'https://www.themoviedb.org/movie/' + str(int(timeoutid))  # + '-' + tt
            response = urllib.request.urlopen(url, timeout=5)
            string = response.read()
            html = string.decode('utf-8')
        except:
            fd = open("timeout1.txt", 'a+')
            fd.write(str(timeoutid)+'\n')
            fd.close()
            count_timeout += 1
    filename = './themoviedb/' + str(timeoutid) + '.txt'
    if html:
        f = open(filename, 'w')
        f.write(html)
        f.close()
    time.sleep(1)
