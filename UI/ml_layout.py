# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'v2.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 680)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 800, 680))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame = QtWidgets.QFrame(self.verticalLayoutWidget)
        self.frame.setEnabled(True)
        self.frame.setStyleSheet("background-image: url(./background.jpeg);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.userid_input = QtWidgets.QLineEdit(self.frame)
        self.userid_input.setGeometry(QtCore.QRect(400, 70, 150, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.userid_input.setFont(font)
        self.userid_input.setObjectName("userid_input")
        self.movieid_input = QtWidgets.QLineEdit(self.frame)
        self.movieid_input.setGeometry(QtCore.QRect(400, 250, 150, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.movieid_input.setFont(font)
        self.movieid_input.setObjectName("movieid_input")
        self.amount_input = QtWidgets.QLineEdit(self.frame)
        self.amount_input.setGeometry(QtCore.QRect(400, 430, 150, 31))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.amount_input.setFont(font)
        self.amount_input.setObjectName("amount_input")
        self.recommend_from_movieid = QtWidgets.QPushButton(self.frame)
        self.recommend_from_movieid.setGeometry(QtCore.QRect(240, 330, 331, 41))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.recommend_from_movieid.setFont(font)
        self.recommend_from_movieid.setAutoDefault(False)
        self.recommend_from_movieid.setDefault(False)
        self.recommend_from_movieid.setFlat(False)
        self.recommend_from_movieid.setObjectName("recommend_from_movieid")
        self.recommend_from_uid = QtWidgets.QPushButton(self.frame)
        self.recommend_from_uid.setGeometry(QtCore.QRect(270, 150, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.recommend_from_uid.setFont(font)
        self.recommend_from_uid.setAutoDefault(False)
        self.recommend_from_uid.setDefault(False)
        self.recommend_from_uid.setFlat(False)
        self.recommend_from_uid.setObjectName("recommend_from_uid")
        self.return_2 = QtWidgets.QPushButton(self.frame)
        self.return_2.setGeometry(QtCore.QRect(0, 0, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.return_2.setFont(font)
        self.return_2.setAutoDefault(False)
        self.return_2.setDefault(False)
        self.return_2.setFlat(False)
        self.return_2.setObjectName("return_2")
        self.verticalLayout_3.addWidget(self.frame)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.amount = QtWidgets.QLabel(self.centralwidget)
        self.amount.setGeometry(QtCore.QRect(240, 430, 110, 40))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.amount.setFont(font)
        self.amount.setObjectName("amount")
        self.movieid = QtWidgets.QLabel(self.centralwidget)
        self.movieid.setGeometry(QtCore.QRect(240, 250, 110, 40))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.movieid.setFont(font)
        self.movieid.setObjectName("movieid")
        self.userid = QtWidgets.QLabel(self.centralwidget)
        self.userid.setGeometry(QtCore.QRect(240, 70, 110, 40))
        font = QtGui.QFont()
        font.setFamily("Consolas")
        font.setPointSize(14)
        self.userid.setFont(font)
        self.userid.setObjectName("userid")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 802, 26))
        self.menubar.setObjectName("menubar")
        self.menusetting = QtWidgets.QMenu(self.menubar)
        self.menusetting.setObjectName("menusetting")
        self.menuhelp = QtWidgets.QMenu(self.menubar)
        self.menuhelp.setObjectName("menuhelp")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionsetting = QtWidgets.QAction(MainWindow)
        self.actionsetting.setObjectName("actionsetting")
        self.actionexit = QtWidgets.QAction(MainWindow)
        self.actionexit.setObjectName("actionexit")
        self.actiondocument_2 = QtWidgets.QAction(MainWindow)
        self.actiondocument_2.setObjectName("actiondocument_2")
        self.actionabout = QtWidgets.QAction(MainWindow)
        self.actionabout.setObjectName("actionabout")
        self.menusetting.addSeparator()
        self.menusetting.addAction(self.actionsetting)
        self.menusetting.addSeparator()
        self.menusetting.addAction(self.actionexit)
        self.menuhelp.addAction(self.actiondocument_2)
        self.menuAbout.addAction(self.actionabout)
        self.menubar.addAction(self.menusetting.menuAction())
        self.menubar.addAction(self.menuhelp.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.userid_input.setText(_translate("MainWindow", "1"))
        self.movieid_input.setText(_translate("MainWindow", "1"))
        self.amount_input.setText(_translate("MainWindow", "10"))
        self.recommend_from_movieid.setText(_translate("MainWindow", "recommend from movieid"))
        self.recommend_from_uid.setText(_translate("MainWindow", "recommend from uid"))
        self.return_2.setText(_translate("MainWindow", "return"))
        self.amount.setText(_translate("MainWindow", "amount"))
        self.movieid.setText(_translate("MainWindow", "movieid"))
        self.userid.setText(_translate("MainWindow", "userid"))
        self.menusetting.setTitle(_translate("MainWindow", "Menu"))
        self.menuhelp.setTitle(_translate("MainWindow", "Help"))
        self.menuAbout.setTitle(_translate("MainWindow", "About"))
        self.actionsetting.setText(_translate("MainWindow", "setting"))
        self.actionexit.setText(_translate("MainWindow", "exit"))
        self.actiondocument_2.setText(_translate("MainWindow", "document"))
        self.actionabout.setText(_translate("MainWindow", "about"))
