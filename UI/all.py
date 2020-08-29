import sys
from PyQt5.QtWidgets import QWidget, QFormLayout, QPushButton, QLabel, QApplication
from PyQt5.QtGui import QFont
from UI import ml_action, cb_action, cf_action
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 登陆对话框
class DialogUI(QWidget):
    def __init__(self, parent=None):
        super(DialogUI, self).__init__(parent)
        self.setWindowTitle("recommendation algorithm")
        self.resize(800, 680)
        self.mylayout()

    def mylayout(self):
        flo = QFormLayout()
        font = QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.btn1 = QPushButton("content based recommend")
        self.btn1.setFont(font)
        self.btn1.clicked.connect(self.jump1)  # 点击取消关闭窗口
        self.btn2 = QPushButton("collaborative filtering")
        self.btn2.setFont(font)
        self.btn2.clicked.connect(self.jump2)  # 点击取消关闭窗口
        self.btn3 = QPushButton("machine learning")
        self.btn3.setFont(font)
        self.btn3.clicked.connect(self.jump3)  # 点击取消关闭窗口
        self.lb = QLabel('')
        for i in range(15):
            flo.addRow(self.lb)
        flo.addRow(self.btn1)
        flo.addRow(self.lb)
        flo.addRow(self.lb)
        flo.addRow(self.btn2)
        flo.addRow(self.lb)
        flo.addRow(self.lb)
        flo.addRow(self.btn3)
        self.setLayout(flo)

    # 跳转
    def jump1(self, text):
        # self.destroy()  # self.close()可以关闭窗体，但窗体还在内存中等待再次显示
        self.close()
        self.content_based_recommend = cb_action.MainCode()  # 生成主窗口的实例
        self.content_based_recommend.show()
        # WindowShow.show()

    # 跳转
    def jump2(self, text):
        self.close()
        self.collaborative_filtering = cf_action.MainCode()  # 生成主窗口的实例
        self.collaborative_filtering.show()

    # 跳转
    def jump3(self, text):
        self.close()
        self.machien_learning = ml_action.MainCode()  # 生成主窗口的实例
        self.machien_learning.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    d = DialogUI()
    d.show()
    sys.exit(app.exec())

