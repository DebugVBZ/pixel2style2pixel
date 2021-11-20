from PyQt5 import QtWidgets
from multiprocessing import Process, Manager, freeze_support
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QSlider,QPushButton,QDesktopWidget,QVBoxLayout,QHBoxLayout,QComboBox,QTextBrowser,QTextEdit,QLabel,QDialog,QFileDialog,QLineEdit,QMessageBox
from PyQt5.QtCore import QCoreApplication,Qt
# from PyQt5 import QtSql
# from PyQt5.QtSql import QSqlQuery
from PyQt5.QtGui import QPixmap,QImage


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
import random

import entrance

# envpath = "/home/melody/anaconda3/envs/python3.7/lib/python3.7/site-packages/cv2/qt/plugins"
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
class ImageProcessUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.srcImageCv = ""
        self.dstImageCv = ""
        self.dstImagePath = ""
        self.processedImagePath = ""
        self.gaussBlurVal = 1
    def initUI(self):
        self.algorithm=''
        self.setGeometry(40,40,800,600)
        
        #语言选择下拉框决定数据库表名及语料库读取


        self.languagelCombo=QComboBox(self,minimumWidth=200) # 语言下拉框
        self.picture = None
    

        # 模糊算法选择器 初始化
        self.languagelCombo.addItem('高斯模糊')
        self.languagelCombo.addItem('均值滤波')
        self.languagelCombo.addItem('双边滤波')
        self.languagelCombo.addItem('2D卷积')

        #模糊前后图片显示Label初始化
        self.srcImageLabel = QLabel("读取源图片")
        self.dstImageLabel = QLabel("模糊后图片")
        self.processedImageLabel = QLabel("复原后图片")

        #参数文本输入框输入
        self.configTestInput = QLineEdit("模糊参数")
        self.configSlider = QSlider(Qt.Horizontal)
        self.configSlider.setMaximum(100)
        self.configSlider.setMinimum(1)
        self.configSlider.setSingleStep(0.1)
        self.configSlider.valueChanged.connect(self.setVal)
        #图片导入按钮 
        self.imageImportBtn = QPushButton("导入图片")
        self.imageImportBtn.clicked.connect(self.loadFile)
        #图片清空按钮
        self.imageClearBtn = QPushButton("图片清空")
        self.imageClearBtn.clicked.connect(self.clearButtonFunc)
        self.imageClearBtn.setEnabled(False)
        #图片模糊按钮
        self.imageBlurringBtn = QPushButton("图片模糊")
        self.imageBlurringBtn.clicked.connect(self.showDstImage)
        self.imageBlurringBtn.setEnabled(False)
        #模糊图片保存按钮
        self.imageSavingBtn = QPushButton("图片保存")
        self.imageSavingBtn.setEnabled(False)
        self.imageSavingBtn.clicked.connect(self.saveFile)

        #图片还原按钮
        self.imageProcessingBtn = QPushButton("图片复原")
        self.imageProcessingBtn.setEnabled(False)
        self.imageProcessingBtn.clicked.connect(self.processingBlurred)

        #初始化GUI布局
        sumVboxLayout = QVBoxLayout()
        #初始化算法选择布局
        algorithmSelectLayout = QVBoxLayout()
        algorithmSelectLayout.addWidget(self.languagelCombo)
        sumVboxLayout.addLayout(algorithmSelectLayout)
        
        #初始化图片显示布局
        imageLayout = QHBoxLayout()
        imageLayout.addWidget(self.srcImageLabel)
        imageLayout.addWidget(self.dstImageLabel)
        imageLayout.addWidget(self.processedImageLabel)
        sumVboxLayout.addLayout(imageLayout)
        #初始化参数配置布局
        configLayout = QVBoxLayout()
        configLayout.addWidget(self.configTestInput)
        configLayout.addWidget(self.configSlider)
        sumVboxLayout.addLayout(configLayout)

        #初始化操作按钮布局
        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.imageImportBtn)
        btnLayout.addWidget(self.imageClearBtn)
        btnLayout.addWidget(self.imageBlurringBtn)
        btnLayout.addWidget(self.imageSavingBtn)
        btnLayout.addWidget(self.imageProcessingBtn)
        sumVboxLayout.addLayout(btnLayout)

        self.setLayout(sumVboxLayout)
    def clearButtonFunc(self):
        self.imageBlurringBtn.setEnabled(False)
        self.imageSavingBtn.setEnabled(False)
        self.srcImageLabel.setPixmap(QPixmap())
        self.dstImageLabel.setPixmap(QPixmap())
        self.processedImageLabel.setPixmap(QPixmap())
        self.imageBlurringBtn.setEnabled(False)
        self.imageSavingBtn.setEnabled(False)
        self.imageClearBtn.setEnabled(False)
        self.imageProcessingBtn.setEnabled(False)
    def loadFile(self):
        print("load--file")
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', './notebooks/images', 'Image files(*.jpg *.gif *.png)')
        self.srcImageCV = cv2.imread(fname)
        self.srcImageCv = cv2.resize(self.srcImageCV, (256, 256))
        if (self.srcImageCV is None):
            return None
        ImageCV2 = cv2.cvtColor(self.srcImageCV,cv2.COLOR_BGR2RGB)
        srcImageCV2 = cv2.resize(ImageCV2,(256,256))
        qImage = QImage(srcImageCV2[:],srcImageCV2.shape[1],srcImageCV2.shape[0],srcImageCV2.shape[1]*3,QImage.Format_RGB888)
        qpixmap = QPixmap(qImage)
        self.imageBlurringBtn.setEnabled(True)
        self.imageClearBtn.setEnabled(True)
        self.srcImageLabel.setPixmap(qpixmap)
    def saveFile(self):
        fileName = QFileDialog.getSaveFileName(self,'Save Image','./',"Image Files (*.jpg)")
        with open(fileName[0],'w') as f:
            cv2.imwrite(fileName[0],self.dstImageCv)
        self.dstImagePath = fileName[0]
        self.imageProcessingBtn.setEnabled(True)
    def convertQImage2CvMap(self,srcImage):
        srcImage = srcImage.converToFormat(QImage.Format.Format_RGBA8888)
        width = srcImage.width()
        height = srcImage.height()
        ptr = srcImage.bits()
        ptr.setSize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return arr
    def showDstImage(self):
        if self.srcImageCV is None:
            return
        img = self.blurring() #
        self.dstImageCv = img
        srcImageCV2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        qImage = QImage(srcImageCV2[:],srcImageCV2.shape[1],srcImageCV2.shape[0],srcImageCV2.shape[1]*3,QImage.Format_RGB888)
        qpixmap = QPixmap(qImage)
        self.dstImageLabel.setPixmap(qpixmap)
        self.imageSavingBtn.setEnabled(True)
    def blurring(self):
        if self.srcImageCV is None:
            return
        image = cv2.resize(self.srcImageCV,(256,256))
        if (self.languagelCombo.currentText()=="高斯模糊"):
            return cv2.GaussianBlur(image, (0, 0), self.gaussBlurVal)
        if (self.languagelCombo.currentText()=="均值滤波"):
            return cv2.blur(image,(self.gaussBlurVal,self.gaussBlurVal))
        if (self.languagelCombo.currentText()=="双边滤波"):
            return self.sp_noise(prob = self.gaussBlurVal/100)
        if (self.languagelCombo.currentText()=="2D卷积"):
            return self.filter2D_demo(image,kernel = self.gaussBlurVal)
    def setVal(self):
        self.gaussBlurVal = self.configSlider.value()
    def filter2D_demo(self,src,kernel):
	# 除以 25 是防止溢出
        kernel = np.ones([kernel,kernel],np.float32)/25
        dst = cv2.filter2D(src,-1,kernel=kernel)
        return dst

    def sp_noise(self,prob):
        image = self.srcImageCV
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output
    def processingBlurred(self):
        self.processedImagePath = entrance.entranceFunction(self.dstImagePath)
        imageCV = cv2.imread(self.processedImagePath)
        ImageCV2 = cv2.cvtColor(imageCV, cv2.COLOR_BGR2RGB)
        srcImageCV2 = cv2.resize(ImageCV2, (256, 256))
        qImage = QImage(srcImageCV2[:], srcImageCV2.shape[1], srcImageCV2.shape[0], srcImageCV2.shape[1] * 3,
                        QImage.Format_RGB888)
        qpixmap = QPixmap(qImage)
        self.processedImageLabel.setPixmap(qpixmap)



if __name__ == "__main__":

    app=QApplication(sys.argv)
    ex=ImageProcessUI()
    ex.show()
    sys.exit(app.exec_())
