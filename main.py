import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pandas as pd
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
        self.filtered_image = None
        self.history = []
        self.redo_stack = []

        self.GaussianButton.clicked.connect(self.GaussianFilter)
        self.EqualizationButton.clicked.connect(self.HistogramEqualization)
        self.HistogramButton.clicked.connect(self.ShowHistogram)

        self.GrayscaleButton.clicked.connect(self.Grayscale)
        self.BinerButton.clicked.connect(self.Biner)
        self.ResetButton.clicked.connect(self.ResetImage)
        self.LoadButton.clicked.connect(self.LoadImage)
        self.SaveButton.clicked.connect(self.SaveImage)
        self.RedoButton.clicked.connect(self.Redo)
        self.UndoButton.clicked.connect(self.Undo)
        self.OriginalExcelButton.clicked.connect(self.SaveOriginalPixeltoExcel)
        self.OriginalTextButton.clicked.connect(self.SaveOriginalPixeltoText)
        self.EditedExcelButton.clicked.connect(self.SaveEditedPixeltoExcel)
        self.EditedTextButton.clicked.connect(self.SaveEditedPixeltoText)

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

    def SaveOriginalPixeltoExcel(self):
        if self.original_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Original to Excel", "", "Excel files (*.xlsx)")
            if file_path:
                rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                df_r = pd.DataFrame(rgb_image[:,:,0])
                df_g = pd.DataFrame(rgb_image[:,:,1])
                df_b = pd.DataFrame(rgb_image[:,:,2])
                
                with pd.ExcelWriter(file_path) as writer:
                    df_r.to_excel(writer, sheet_name='Red Channel')
                    df_g.to_excel(writer, sheet_name='Green Channel')
                    df_b.to_excel(writer, sheet_name='Blue Channel')

    def SaveEditedPixeltoExcel(self):
        if self.Image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Edited to Excel", "", "Excel files (*.xlsx)")
            if file_path:
                rgb_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
                df_r = pd.DataFrame(rgb_image[:,:,0])
                df_g = pd.DataFrame(rgb_image[:,:,1])
                df_b = pd.DataFrame(rgb_image[:,:,2])
                
                with pd.ExcelWriter(file_path) as writer:
                    df_r.to_excel(writer, sheet_name='Red Channel')
                    df_g.to_excel(writer, sheet_name='Green Channel')
                    df_b.to_excel(writer, sheet_name='Blue Channel')

    def SaveOriginalPixeltoText(self):
        if self.original_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Original to Text", "", "Text files (*.txt)")
            if file_path:
                with open(file_path, 'w') as f:
                    rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    height, width = rgb_image.shape[:2]
                    
                    f.write("Original Image Pixel Values (RGB):\n")
                    for i in range(height):
                        for j in range(width):
                            pixel = rgb_image[i,j]
                            f.write(f"Pixel [{i},{j}]: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}\n")

    def SaveEditedPixeltoText(self):
        if self.Image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Edited to Text", "", "Text files (*.txt)")
            if file_path:
                with open(file_path, 'w') as f:
                    rgb_image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
                    height, width = rgb_image.shape[:2]
                    
                    f.write("Edited Image Pixel Values (RGB):\n")
                    for i in range(height):
                        for j in range(width):
                            pixel = rgb_image[i,j]
                            f.write(f"Pixel [{i},{j}]: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}\n")

    def Undo(self):
        if len(self.history) > 1:
            self.redo_stack.append(self.history.pop())
            self.filtered_image = self.history[-1].copy()
            self.AdjustImage()

    def Redo(self):
        if self.redo_stack:
            self.history.append(self.redo_stack.pop())
            self.filtered_image = self.history[-1].copy()
            self.AdjustImage()

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
            self.Image = self.original_image.copy()
            self.history = [self.original_image.copy()]
            self.redo_stack = []
            self.displayImage(1)
            self.AdjustImage()

    def SaveImage(self):
        if self.Image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.Image)

    def ResetImage(self):
        if self.original_image is not None:
            self.filtered_image = self.original_image.copy()
            self.history = [self.original_image.copy()]
            self.redo_stack = []
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
        self.history.append(self.filtered_image.copy())
        self.redo_stack = []
        self.AdjustImage()

    def Biner(self):
        gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        self.filtered_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self.history.append(self.filtered_image.copy())
        self.redo_stack = []
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
        self.history.append(self.filtered_image.copy())
        self.redo_stack = []
        self.AdjustImage()

    def HistogramEqualization(self):
        hist, bins = np.histogram(self.filtered_image.flatten(), 256, [0, 256]) 
        cdf = hist.cumsum() 
        cdf_normalized = cdf * hist.max() / cdf.max() 
        cdf_m = np.ma.masked_equal(cdf, 0) 
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) 
        cdf = np.ma.filled(cdf_m, 0).astype('uint8') 
        self.filtered_image = cdf[self.filtered_image] 
        self.history.append(self.filtered_image.copy())
        self.redo_stack = []
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.filtered_image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()
        self.AdjustImage()

    def ShowHistogram(self):
        hist, bins = np.histogram(self.filtered_image.flatten(), 256, [0, 256])
        plt.hist(self.filtered_image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
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