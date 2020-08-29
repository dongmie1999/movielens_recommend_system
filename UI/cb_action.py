import sys
import csv
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication
from PyQt5.QtCore import QCoreApplication
from content_based_recommend import main
from UI import all
from UI import cb_layout


class MainCode(QMainWindow, cb_layout.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        QMainWindow.__init__(self)
        cb_layout.Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setWindowTitle("content based recommend")
        # self.userid_input.activated[str].connect(self.set_uid)
        # self.movieid_input.activated[str].connect(self.set_mid)
        self.recommend_from_uid.clicked.connect(self.r_uid)
        self.return_2.clicked.connect(self.return_to)
        # 以下是菜单栏的选项
        self.actiondocument_2.triggered.connect(self.document)
        self.actionsetting.triggered.connect(self.setting)
        self.actionabout.triggered.connect(self.about)
        self.actionexit.triggered.connect(QCoreApplication.instance().quit)
        # 为检查输入的用户id是否在数据集里
        with open('../content_based_recommend/data/ratings.csv') as csvfile:
            reader = csv.reader(csvfile)
            column = [row[0] for row in reader]
            self.uid_in_dataset = set(column[1:])
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
                print(main.content_based_recommend(self.uid, self.amount).main())
                print("done")
            else:
                text = "amount should be a positive number"
                QMessageBox.information(self, "Message", text, QMessageBox.Ok)
        else:
            text = "userid should be included in dataset"
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

    def go(self):
        try:
            self.bpm = int(self.w_bpm.text())
        except ValueError:
            text = "bpm should be an positive integer.\nrecommend: 70~150"
            QMessageBox.information(self, "Message", text, QMessageBox.Ok)
            return
        try:
            if not 0 < int(self.w_repeat.text()) < 20:
                raise ValueError
        except ValueError:
            text = "repeat should be an positive integer.\nrecommend: 1~5"
            QMessageBox.information(self, "Message", text, QMessageBox.Ok)
            return
        self.intensity = int(self.w_intensity.value()/100)
        try:  # 是用级数表示的和弦
            for t in self.w_chord_progression.text():
                if 0 < int(t) < 8:
                    pass
                else:
                    text = "Input should be a series fo numbers.\nEach number must be between 1~7.\n" + \
                           "Example: 4321 or 4536251 or 1645"
                    QMessageBox.information(self, "Message", text, QMessageBox.Ok)
                    return
            self.chord_progression = self.w_chord_progression.text()
        except ValueError:  # 是和弦名称
            self.chord_progression = self.w_chord_progression.text().split(',')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())
