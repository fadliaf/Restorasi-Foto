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
        self.original_image = None  # Simpan gambar asli agar tidak terpengaruh saat mengedit
        
        # Fungsi button filter
        self.GaussianButton.clicked.connect(self.GaussianFilter)
        self.HistogramEqualizationButton.clicked.connect(self.HistogramEqualization)
        self.pushButton_5.clicked.connect(self.ShowHistogramEqualization)
        # Fungsi button 
        self.GrayscaleButton.clicked.connect(self.Grayscale)
        self.BinerButton.clicked.connect(self.Biner)
        self.ResetButton.clicked.connect(self.ResetImage)
        self.LoadButton.clicked.connect(self.LoadImage)
        self.SaveButton.clicked.connect(self.SaveImage)

        # Fungsi slider
        self.BrightnessSlider.valueChanged.connect(self.AdjustImage)
        self.ContrastSlider.valueChanged.connect(self.AdjustImage)
        self.SharpeningSlider.valueChanged.connect(self.AdjustImage)
        self.SaturationSlider.valueChanged.connect(self.AdjustImage)
        self.HueSlider.valueChanged.connect(self.AdjustImage)
        self.ValueSlider.valueChanged.connect(self.AdjustImage)
        
        # Set nilai awal dan rentang slider
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

            # **🔹 Resize gambar jika terlalu besar**
            max_width, max_height = 600, 350
            height, width = self.original_image.shape[:2]

            if width > max_width or height > max_height:
                # Hitung rasio skala untuk menjaga aspect ratio
                scale = min(max_width / width, max_height / height)
                new_size = (int(width * scale), int(height * scale))
                self.original_image = cv2.resize(self.original_image, new_size, interpolation=cv2.INTER_AREA)

            self.Image = self.original_image.copy()
            self.displayImage()

    def SaveImage(self):
        if self.Image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Simpan Gambar", "", "Images (*.png *.jpg *.bmp)")
            if file_path:
                cv2.imwrite(file_path, self.Image)
                
    def ResetImage(self):
        if self.original_image is not None:
            self.Image = self.original_image.copy()
            self.displayImage(2)
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
        H, W = self.Image.shape[:2]
        gray = np.zeros((H,W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i,j] = np.clip(0.299*self.Image[i,j,0] + 
                                    0.587*self.Image[i,j,1] + 
                                    0.114*self.Image[i,j,2], 0, 255)

        self.Image = gray
        self.displayImage(2)
        
    def Biner(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]

        for i in range(H):
            for j in range(W):
                a = self.Image[i, j]
                if a == 180:
                    b = 1
                elif a < 180:
                    b = 0
                elif a > 180:
                    b = 255
                self.Image[i, j] = b
                
        self.displayImage(2)

    def Convolve(self, kernel):
        # Get image dimensions
        img_height, img_width = self.Image.shape[:2]
        
        # Get kernel dimensions 
        kernel_height, kernel_width = kernel.shape
        
        # Calculate half sizes
        H = kernel_height // 2
        W = kernel_width // 2
        
        # Create output Image
        output = np.zeros_like(self.Image)
        
        # Perform convolution
        for i in range(H+1, img_height-H):
            for j in range(W+1, img_width-W):
                # For each color channel
                for c in range(3):
                    sum = 0
                    for k in range(-H, H+1):
                        for l in range(-W, W+1):
                            a = self.Image[i+k, j+l, c]
                            w = kernel[H+k, W+l]
                            sum += w * a
                    output[i, j, c] = sum
                    
        return output

    def GaussianFilter(self):
        # Create Gaussian kernel using the formula:
        # G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2 + y^2)/(2*sigma^2))
        kernel_size = 3
        sigma = 1.0
        
        # Create coordinate matrices
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        
        # Calculate Gaussian kernel values
        kernel = (1/(2*np.pi*sigma**2)) * np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply convolution using the existing convolve method
        filtered_image = self.Convolve(kernel)
        
        # Store pixel values before convolution for analysis
        original_pixels = self.Image.copy()
        
        # Update image with filtered result
        self.Image = filtered_image
        
        # Display filtered image
        self.displayImage(2)

    def HistogramEqualization(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(2)
        
        return cdf_normalized

    def ShowHistogramEqualization(self):
        # Get the normalized CDF by calling HistogramEqualization
        cdf_normalized = self.HistogramEqualization()
        
        # Plot CDF and histogram
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def AdjustImage(self):
        if self.original_image is None:
            return

        # Ambil nilai dari slider
        brightness = self.BrightnessSlider.value()
        contrast = self.ContrastSlider.value() / 100.0
        sharpening = self.SharpeningSlider.value()
        saturation = self.SaturationSlider.value() / 100.0
        hue_shift = self.HueSlider.value()
        value_scale = self.ValueSlider.value() / 100.0

        # Salin gambar asli
        adjusted_image = self.original_image.copy()

        # 🔹 BRIGHTNESS & CONTRAST
        adjusted_image = cv2.convertScaleAbs(adjusted_image, alpha=contrast, beta=brightness)

        # 🔹 SHARPENING
        if sharpening > 0:
            kernel = np.array([[-1, -1, -1],
                            [-1, 9 + sharpening, -1],
                            [-1, -1, -1]], dtype=np.float32)
            adjusted_image = cv2.filter2D(adjusted_image, -1, kernel)

        # 🔹 SATURATION, HUE, VALUE
        hsv = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Saturasi
        hsv[..., 1] *= saturation

        # Hue (geser dan pastikan dalam rentang 0-179)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

        # Value
        hsv[..., 2] *= value_scale

        # Clipping agar tetap valid
        hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

        # Konversi kembali ke BGR
        adjusted_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Simpan dan tampilkan
        self.Image = adjusted_image
        self.displayImage(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Pengolahan Citra Digital')
    window.show()
    sys.exit(app.exec_())
