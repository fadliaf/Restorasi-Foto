import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('tubes.ui', self)
        self.Image = None
        self.original_image = None
        self.filtered_image = None  # Untuk menyimpan hasil setelah filter sebelum penyesuaian

        self.GaussianButton.clicked.connect(self.GaussianFilter)
        self.EqualizationButton.clicked.connect(self.HistogramEqualization)
        self.pushButton_5.clicked.connect(self.ShowHistogramEqualization)

        self.GrayscaleButton.clicked.connect(self.Grayscale)
        self.BinerButton.clicked.connect(self.Biner)
        self.ResetButton.clicked.connect(self.ResetImage)
        self.LoadButton.clicked.connect(self.LoadImage)
        self.SaveButton.clicked.connect(self.SaveImage)

        self.BrightnessSlider.valueChanged.connect(self.AdjustImage)
        self.ContrastSlider.valueChanged.connect(self.AdjustImage)
        self.SharpeningSlider.valueChanged.connect(self.AdjustImage)
        self.SaturationSlider.valueChanged.connect(self.AdjustImage)
        self.HueSlider.valueChanged.connect(self.AdjustImage)
        self.ValueSlider.valueChanged.connect(self.AdjustImage)

        self.BrightnessSlider.setRange(-100, 100)
        self.BrightnessSlider.setValue(0)
        self.ContrastSlider.setRange(0, 200)
        self.ContrastSlider.setValue(100)
        self.SharpeningSlider.setRange(0, 10)
        self.SharpeningSlider.setValue(0)
        self.SaturationSlider.setRange(0, 200)
        self.SaturationSlider.setValue(100)
        self.HueSlider.setRange(-180, 180)
        self.HueSlider.setValue(0)
        self.ValueSlider.setRange(0, 200)
        self.ValueSlider.setValue(100)

    def LoadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            self.original_image = cv2.imread(file_path)
            max_width, max_height = 600, 350
            height, width = self.original_image.shape[:2]
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_size = (int(width * scale), int(height * scale))
                self.original_image = cv2.resize(self.original_image, new_size, interpolation=cv2.INTER_AREA)
            self.filtered_image = self.original_image.copy()
            self.AdjustImage()

    def SaveImage(self):
        if self.Image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.Image)

    def ResetImage(self):
        if self.original_image is not None:
            self.filtered_image = self.original_image.copy()
            self.AdjustImage()
            self.BrightnessSlider.setValue(0)
            self.ContrastSlider.setValue(100)
            self.SharpeningSlider.setValue(0)
            self.SaturationSlider.setValue(100)
            self.HueSlider.setValue(0)
            self.ValueSlider.setValue(100)

    def displayImage(self, windows=1):
        if self.Image is None:
            return
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qformat = QImage.Format_RGB888
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
        if windows == 1:
            self.InputImage.setPixmap(QPixmap.fromImage(qimg))
            self.InputImage.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        else:
            self.OutputImage.setPixmap(QPixmap.fromImage(qimg))
            self.OutputImage.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def Grayscale(self):
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.AdjustImage()

    def Biner(self):
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        self.filtered_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.AdjustImage()

    def Convolve(self, kernel):
        return cv2.filter2D(self.filtered_image, -1, kernel)

    def GaussianFilter(self):
        kernel_size = 3
        sigma = 1.0
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = (1/(2*np.pi*sigma**2)) * np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel = kernel / np.sum(kernel)
        self.filtered_image = self.Convolve(kernel)
        self.AdjustImage()

    def HistogramEqualization(self):
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        self.filtered_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        self.AdjustImage()

    def ShowHistogramEqualization(self):
        self.HistogramEqualization()
        hist, bins = np.histogram(cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY).flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.filtered_image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def AdjustImage(self):
        if self.filtered_image is None:
            return

        image = self.filtered_image.copy()

        brightness = self.BrightnessSlider.value()
        contrast = self.ContrastSlider.value() / 100.0
        sharpening = self.SharpeningSlider.value()
        saturation = self.SaturationSlider.value() / 100.0
        hue_shift = self.HueSlider.value()
        value_scale = self.ValueSlider.value() / 100.0

        image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

        if sharpening > 0:
            kernel = np.array([[-1, -1, -1], [-1, 9 + sharpening, -1], [-1, -1, -1]], dtype=np.float32)
            image = cv2.filter2D(image, -1, kernel)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= saturation
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 2] *= value_scale
        hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        self.Image = image
        self.displayImage(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Pengolahan Citra Digital')
    window.show()
    sys.exit(app.exec_())