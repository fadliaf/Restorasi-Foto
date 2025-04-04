import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
import numpy as np 
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage,self).__init__()
        loadUi('tubes.ui', self)

app = QtWidgets.QApplication(sys.argv) 
window = ShowImage() 
window.setWindowTitle('Pengolahan Citra Digital') 
window.show() 
sys.exit(app.exec_())