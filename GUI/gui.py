# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'test.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsView, QGridLayout,
    QLCDNumber, QLabel, QMainWindow, QProgressBar,
    QSizePolicy, QStatusBar, QVBoxLayout, QWidget)

class Ui_Sensor(object):
    def setupUi(self, Sensor):
        if not Sensor.objectName():
            Sensor.setObjectName(u"Sensor")
        self.setStyleSheet('background-color : white;')
        Sensor.resize(831, 836)
        self.centralwidget = QWidget(Sensor)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(280, 30, 531, 171))
        self.heart_rate_graph = QVBoxLayout(self.verticalLayoutWidget)
        self.heart_rate_graph.setObjectName(u"heart_rate_graph")
        self.heart_rate_graph.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(280, 220, 531, 171))
        self.resp_rate_graph = QVBoxLayout(self.verticalLayoutWidget_2)
        self.resp_rate_graph.setObjectName(u"resp_rate_graph")
        self.resp_rate_graph.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutWidget_3 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(280, 410, 531, 171))
        self.conductivity_graph = QVBoxLayout(self.verticalLayoutWidget_3)
        self.conductivity_graph.setObjectName(u"conductivity_graph")
        self.conductivity_graph.setContentsMargins(0, 0, 0, 0)
        self.heart_rate_text = QLabel(self.centralwidget)
        self.heart_rate_text.setObjectName(u"heart_rate_text")
        self.heart_rate_text.setGeometry(QRect(10, 130, 181, 51))
        font = QFont()
        font.setFamilies([u"Sans"])
        font.setPointSize(16)
        font.setBold(True)
        self.heart_rate_text.setFont(font)
        self.heart_rate_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heart_rate_gif = QGraphicsView(self.centralwidget)
        self.heart_rate_gif.setObjectName(u"heart_rate_gif")
        self.heart_rate_gif.setGeometry(QRect(20, 30, 231, 91))
        self.heart_rate_gif.setStyleSheet(u"border: none;\n"
"")
        self.heart_rate_number = QLCDNumber(self.centralwidget)
        self.heart_rate_number.setObjectName(u"heart_rate_number")
        self.heart_rate_number.setGeometry(QRect(130, 120, 151, 61))
        self.heart_rate_number.setStyleSheet(u"border: none;\n"
"")
        self.heart_rate_number.setSmallDecimalPoint(False)
        self.heart_rate_number.setDigitCount(3)
        self.heart_rate_number.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.conductivity_text = QLabel(self.centralwidget)
        self.conductivity_text.setObjectName(u"conductivity_text")
        self.conductivity_text.setGeometry(QRect(30, 540, 201, 51))
        font1 = QFont()
        font1.setFamilies([u"Sans"])
        font1.setPointSize(16)
        font1.setBold(True)
        self.conductivity_text.setFont(font1)
        self.conductivity_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resp_rate_text = QLabel(self.centralwidget)
        self.resp_rate_text.setObjectName(u"resp_rate_text")
        self.resp_rate_text.setGeometry(QRect(10, 320, 181, 51))
        self.resp_rate_text.setFont(font)
        self.resp_rate_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resp_rate_number = QLCDNumber(self.centralwidget)
        self.resp_rate_number.setObjectName(u"resp_rate_number")
        self.resp_rate_number.setGeometry(QRect(130, 310, 151, 61))
        self.resp_rate_number.setStyleSheet(u"border: none;\n"
"")
        self.resp_rate_number.setSmallDecimalPoint(False)
        self.resp_rate_number.setDigitCount(3)
        self.resp_rate_number.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.resp_rate_gif = QGraphicsView(self.centralwidget)
        self.resp_rate_gif.setObjectName(u"resp_rate_gif")
        self.resp_rate_gif.setGeometry(QRect(20, 220, 231, 91))
        self.resp_rate_gif.setStyleSheet(u"border: none;\n"
"")
        self.line = QFrame(self.centralwidget)
        self.line.setObjectName(u"line")
        self.line.setGeometry(QRect(10, 580, 801, 16))
        self.line.setFrameShadow(QFrame.Shadow.Plain)
        self.line.setLineWidth(3)
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.conductivity_progressbar = QProgressBar(self.centralwidget)
        self.conductivity_progressbar.setObjectName(u"conductivity_progressbar")
        self.conductivity_progressbar.setGeometry(QRect(30, 430, 211, 91))
        self.conductivity_progressbar.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.conductivity_progressbar.setStyleSheet(u"QProgressBar {\n"
                        
"                background: qlineargradient(\n"
"                     x1:0, y1:0, x2:1, y2:0,\n"
"                    stop:0 #00FF00,   \n"
"                    stop:0.5 #FFFF00, \n"
"                    stop:1 #FF0000    \n"
"                );\n"
"            }\n"
"            QProgressBar::chunk {\n"
"                background-color:#FFFFFF;\n"
"                \n"
"            }")
        self.conductivity_progressbar.setValue(24)
        self.conductivity_progressbar.setTextVisible(False)
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(20, 200, 791, 21))
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_3 = QFrame(self.centralwidget)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(20, 390, 791, 21))
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_14 = QFrame(self.centralwidget)
        self.line_14.setObjectName(u"line_14")
        self.line_14.setGeometry(QRect(252, 20, 21, 561))
        self.line_14.setFrameShadow(QFrame.Shadow.Raised)
        self.line_14.setFrameShape(QFrame.Shape.VLine)
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(15, 615, 801, 171))
        self.gridLayout = QGridLayout(self.widget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.presence_text = QLabel(self.widget)
        self.presence_text.setObjectName(u"presence_text")
        self.presence_text.setFont(font1)
        self.presence_text.setFrameShape(QFrame.Shape.NoFrame)
        self.presence_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.presence_text.setMargin(0)
        self.presence_text.setIndent(-1)

        self.gridLayout.addWidget(self.presence_text, 0, 0, 1, 1)

        self.drowsiness_text = QLabel(self.widget)
        self.drowsiness_text.setObjectName(u"drowsiness_text")
        self.drowsiness_text.setFont(font1)
        self.drowsiness_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.drowsiness_text, 0, 1, 1, 1)

        self.grip_text = QLabel(self.widget)
        self.grip_text.setObjectName(u"grip_text")
        self.grip_text.setFont(font1)
        self.grip_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.grip_text, 0, 2, 1, 1)

        self.warning_text = QLabel(self.widget)
        self.warning_text.setObjectName(u"warning_text")
        self.warning_text.setFont(font1)
        self.warning_text.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.warning_text, 0, 3, 1, 1)

        self.presence_image = QLabel(self.widget)
        self.presence_image.setObjectName(u"presence_image")
        self.presence_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.presence_image, 1, 0, 1, 1)

        self.drowsiness_image = QLabel(self.widget)
        self.drowsiness_image.setObjectName(u"drowsiness_image")
        self.drowsiness_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.drowsiness_image, 1, 1, 1, 1)

        self.grip_image = QLabel(self.widget)
        self.grip_image.setObjectName(u"grip_image")
        self.grip_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.grip_image, 1, 2, 1, 1)

        self.warning_image = QLabel(self.widget)
        self.warning_image.setObjectName(u"warning_image")
        self.warning_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.warning_image, 1, 3, 1, 1)

        Sensor.setCentralWidget(self.centralwidget)
        self.verticalLayoutWidget.raise_()
        self.verticalLayoutWidget_2.raise_()
        self.verticalLayoutWidget_3.raise_()
        self.heart_rate_gif.raise_()
        self.conductivity_text.raise_()
        self.resp_rate_number.raise_()
        self.resp_rate_gif.raise_()
        self.conductivity_progressbar.raise_()
        self.heart_rate_number.raise_()
        self.line_14.raise_()
        self.line_3.raise_()
        self.line_2.raise_()
        self.line.raise_()
        self.heart_rate_text.raise_()
        self.resp_rate_text.raise_()
        self.statusbar = QStatusBar(Sensor)
        self.statusbar.setObjectName(u"statusbar")
        Sensor.setStatusBar(self.statusbar)

        self.retranslateUi(Sensor)

        QMetaObject.connectSlotsByName(Sensor)
    # setupUi

    def retranslateUi(self, Sensor):
        Sensor.setWindowTitle(QCoreApplication.translate("Sensor", u"MainWindow", None))
        self.heart_rate_text.setText(QCoreApplication.translate("Sensor", u"Heart _Rate: ", None))
        self.conductivity_text.setText(QCoreApplication.translate("Sensor", u"Conductivity(%)", None))
        self.resp_rate_text.setText(QCoreApplication.translate("Sensor", u"Resp _Rate: ", None))
        self.presence_text.setText(QCoreApplication.translate("Sensor", u"Presence", None))
        self.drowsiness_text.setText(QCoreApplication.translate("Sensor", u"Drowsiness", None))
        self.grip_text.setText(QCoreApplication.translate("Sensor", u"Safety_grip", None))
        self.warning_text.setText(QCoreApplication.translate("Sensor", u"Warning", None))
        self.presence_image.setText(QCoreApplication.translate("Sensor", u"TextLabel", None))
        self.drowsiness_image.setText(QCoreApplication.translate("Sensor", u"TextLabel", None))
        self.grip_image.setText(QCoreApplication.translate("Sensor", u"TextLabel", None))
        self.warning_image.setText(QCoreApplication.translate("Sensor", u"TextLabel", None))
    # retranslateUi

