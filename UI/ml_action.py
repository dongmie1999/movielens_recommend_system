import sys
import pandas as pd
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication
from PyQt5.QtCore import QCoreApplication
from machine_learning import predict
from UI import all
from UI import ml_layout
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class MainCode(QMainWindow, ml_layout.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        QMainWindow.__init__(self)
        ml_layout.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("machine learning")
        # self.userid_input.activated[str].connect(self.set_uid)
        # self.movieid_input.activated[str].connect(self.set_mid)
        self.recommend_from_uid.clicked.connect(self.r_uid)
        self.recommend_from_movieid.clicked.connect(self.r_mid)
        self.return_2.clicked.connect(self.return_to)
        # 以下是菜单栏的选项
        self.actiondocument_2.triggered.connect(self.document)
        self.actionsetting.triggered.connect(self.setting)
        self.actionabout.triggered.connect(self.about)
        self.actionexit.triggered.connect(QCoreApplication.instance().quit)
        self.uid_in_dataset = set()  # 数据集里的用户id
        with open('../machine_learning/ml-1m/ratings.dat', 'r') as f:
            for line in f:
                s = line.strip().split('::')
                self.uid_in_dataset.add(s[0])
        movies_title = ['MovieID', 'Title', 'Genres']
        movies = pd.read_csv('../machine_learning/ml-1m/movies.dat', sep='::', header=None, names=movies_title,
                             engine='python')
        self.mid_in_dataset = set(movies['MovieID'])  # 数据集里的电影id
        self.num = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']  # 检查输入的数量是否符合要求

    # 跳转
    def return_to(self):
        self.close()
        self.menu = all.DialogUI()
        self.menu.show()

    def r_uid(self):
        if self.userid_input.text() in self.uid_in_dataset:
            # 检查数量的输入是否符合要求
            isnum = True
            for t in self.amount_input.text():
                if t not in self.num:
                    isnum = False
                    break
            try:
                self.amount = int(self.amount_input.text())
            except ValueError:
                isnum = False
            if isnum and self.amount > 0:  # 数量是正整数的话
                self.uid = int(self.userid_input.text())
                print("ready")
                print(predict.recommend_your_favorite_movie(user_id_val=self.uid, top_k=self.amount))
                print("done")
            else:
                text = "amount should be a positive number"
                QMessageBox.information(self, "Message", text, QMessageBox.Ok)
        else:
            text = "userid should be included in dataset"
            QMessageBox.information(self, "Message", text, QMessageBox.Ok)

    def r_mid(self):
        if self.movieid_input.text() in self.uid_in_dataset:
            # 检查数量的输入是否符合要求
            isnum = True
            for t in self.amount_input.text():
                if t not in self.num:
                    isnum = False
                    break
            try:
                self.amount = int(self.amount_input.text())
            except ValueError:
                isnum = False
            if isnum and self.amount > 0:  # 数量是正整数的话
                self.mid = int(self.movieid_input.text())
                print("ready")
                print(predict.recommend_same_type_movie(movie_id_val=self.mid, top_k=self.amount))
                print("done")
            else:
                text = "amount should be a positive number"
                QMessageBox.information(self, "Message", text, QMessageBox.Ok)
        else:
            text = "movieid should be included in dataset"
            QMessageBox.information(self, "Message", text, QMessageBox.Ok)

    def document(self):
        text = "this is ducoment"
        QMessageBox.information(self, "Message", text, QMessageBox.Ok)

    def setting(self):
        text = "this is setting"
        QMessageBox.information(self, "Message", text, QMessageBox.Ok)

    def about(self):
        text = "author: @dongmie1999\n2020.4"
        QMessageBox.information(self, "Message", text, QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
