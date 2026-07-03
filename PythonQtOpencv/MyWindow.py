import cv2
import sys
import numpy as np
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QMainWindow
from PythonQtOpencvUI import Ui_MainWindow
from MyMatImage import MyMatImage

class MyWindow(QMainWindow, Ui_MainWindow, MyMatImage):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        # 文件菜单
        self.actionOpenFile.triggered.connect(self.openFile)
        self.actionRecovery.triggered.connect(self.recovery)
        self.actionClear.triggered.connect(self.clear)
        self.actionClose.triggered.connect(self.close)

        # 图像处理
        self.actionGray.triggered.connect(self.gray)
        self.actionBlur.triggered.connect(self.blur)
        self.actionCanny.triggered.connect(self.canny)
        self.actionThreshold.triggered.connect(self.threshold)

        # 人脸检测
        self.actionFaceDetect.triggered.connect(self.faceDetect)

        # 图像检索（你最关心的）
        self.actionImageRetrieval.triggered.connect(self.imageRetrieval)

        # 视频
        self.actionOpenCamera.triggered.connect(self.openCamera)

        self.cap = None
        self.timer = None

    # -------------------- 文件操作 --------------------
    def openFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, "打开图片", "C:/",
                                               "Images (*.bmp *.jpg *.jpeg *.png *.txt)")
        if not fname:
            return

        src = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
        MyMatImage.srcImage = src
        self.showImage(src, self.SourceImageLabel)

    def recovery(self):
        if hasattr(MyMatImage, 'srcImage') and MyMatImage.srcImage is not None:
            self.showImage(MyMatImage.srcImage, self.TargetImageLabel)

    def clear(self):
        self.SourceImageLabel.clear()
        self.TargetImageLabel.clear()
        self.SourceImageLabel.setText("Source Image")
        self.TargetImageLabel.setText("Result Image")

    # -------------------- 工具函数 --------------------
    def showImage(self, img, label):
        if img is None:
            return
        if len(img.shape) == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        h, w = rgb.shape[:2]
        qimg = QtGui.QImage(rgb.data, w, h, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # -------------------- 图像处理 --------------------
    def gray(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开图片！")
            return
        gray = cv2.cvtColor(MyMatImage.srcImage, cv2.COLOR_BGR2GRAY)
        self.showImage(gray, self.TargetImageLabel)

    def blur(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开图片！")
            return
        blur = cv2.GaussianBlur(MyMatImage.srcImage, (5, 5), 0)
        self.showImage(blur, self.TargetImageLabel)

    def canny(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开图片！")
            return
        gray = cv2.cvtColor(MyMatImage.srcImage, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 50, 150)
        self.showImage(canny, self.TargetImageLabel)

    def threshold(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开图片！")
            return
        gray = cv2.cvtColor(MyMatImage.srcImage, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        self.showImage(th, self.TargetImageLabel)

    # -------------------- 人脸检测 --------------------
    def faceDetect(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开图片！")
            return

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(MyMatImage.srcImage, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        img = MyMatImage.srcImage.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        self.showImage(img, self.TargetImageLabel)

    # -------------------- 图像检索（核心） --------------------
    def imageRetrieval(self):
        if not hasattr(MyMatImage, 'srcImage') or MyMatImage.srcImage is None:
            QMessageBox.warning(self, "提示", "请先打开待检索图片！")
            return

        dir_path = QFileDialog.getExistingDirectory(self, "选择图库文件夹", "C:/")
        if not dir_path:
            return

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(MyMatImage.srcImage, None)
        if des1 is None:
            QMessageBox.warning(self, "提示", "无法提取特征！")
            return

        bf = cv2.BFMatcher()
        best_img = None
        best_cnt = 0

        for name in os.listdir(dir_path):
            path = os.path.join(dir_path, name)
            if not name.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                continue

            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
            if img is None:
                continue

            kp2, des2 = sift.detectAndCompute(img, None)
            if des2 is None:
                continue

            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) > best_cnt:
                best_cnt = len(good)
                best_img = img

        if best_img is not None:
            self.showImage(best_img, self.TargetImageLabel)
            QMessageBox.information(self, "完成", f"匹配到 {best_cnt} 个特征点")
        else:
            QMessageBox.information("提示", "未找到相似图片")

    # -------------------- 摄像头 --------------------
    def openCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def updateFrame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.showImage(frame, self.SourceImageLabel)
        else:
            self.timer.stop()
            self.cap.release()