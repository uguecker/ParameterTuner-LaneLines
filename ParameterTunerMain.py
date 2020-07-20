
# --------------------------
import sys
import os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.image as mpimg
import cv2

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# -------------------------
# local imports
from ImageProcessingPipeline import CLaneFindingPipeline

# =================================================
# Matplotlib Canvas
# =================================================
class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
# -----------------------------------

# =================================================
# MainWindow
# =================================================
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        
    
        # Tuning parameters
        self.set_parameter_default_values()
        
        # ------------------------------------
        # Configure MainWindow
        
        # WindowTitle
        self.setWindowTitle("Analyzer V0.1")

        # Create toolbar and statusbar
        self.Create_Toolbar_Statusbar()
        
        # Layout 
        self.central_widget = QWidget()
                              
        self.layoutOuter = QVBoxLayout(self.central_widget)
        #self.layoutOuter.setContentsMargins(0,0,0,0)
        #self.layoutOuter.setSpacing(0)
                
        self.layoutOuter.addLayout(self.Create_FiguresLayout())
        self.layoutOuter.addLayout(self.Create_SlidersLayout())
               
        self.setCentralWidget(self.central_widget)
        
        
        self.iLaneFindingPipeline = CLaneFindingPipeline()
        
        # refresh
        self.update()
        
    # ---------------------------
    def Create_SlidersLayout(self):
        # Slider1: gaussian_blur
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(11)
        slider.setValue(self.gaussian_blur_kernel_size)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
        slider.valueChanged.connect(self.update_gaussian_blur_kernel_size)
        self.slider1 = slider
        
        # Slider2: canny_low_threshold
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(self.canny_low_threshold)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
        slider.valueChanged.connect(self.update_canny_low_threshold)
        self.slider2 = slider
        
        
        # Slider3: canny_high_threshold
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(250)
        slider.setValue(self.canny_high_threshold)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(2)
        slider.valueChanged.connect(self.update_canny_high_threshold)
        self.slider3 = slider
        
        # Slider4: hough_rho
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(20)
        slider.setValue(round(self.hough_rho*10.0))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.update_hough_rho)
        self.slider4 = slider
        
        
        # Slider5: hough_theta
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(20)
        slider.setValue(round(self.hough_theta*10.0))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.update_hough_theta)
        self.slider5 = slider
        
        
        # Slider6: hough_threshold
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(20)
        slider.setValue(self.hough_threshold)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.update_hough_threshold)
        self.slider6 = slider
        
        # Slider7: hough_min_line_len
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(self.hough_min_line_len)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.update_hough_min_line_len)
        self.slider7 = slider
        
        # Slider7: hough_max_line_gap
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(250)
        slider.setValue(self.hough_max_line_gap)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(self.update_hough_max_line_gap)
        self.slider8 = slider
        
        self.label_gaussian_blur_kernel_size = QLabel('??')
        self.label_canny_low_threshold       = QLabel('??')
        self.label_canny_high_threshold      = QLabel('??')
        self.label_hough_rho                 = QLabel('??')
        self.label_hough_theta               = QLabel('??')
        self.label_hough_threshold           = QLabel('??')
        self.label_hough_min_line_len        = QLabel('??')
        self.label_hough_max_line_gap        = QLabel('??')
        
        
        
        # Button - Reset - Set tuning parameters to default values
        self.pushButton_reset_parameters = QPushButton('Reset')
        self.pushButton_reset_parameters.setToolTip('Set tuning parameters to default values')
        self.pushButton_reset_parameters.clicked.connect(self.on_click_reset_parametrs)
  
        
        # Grid
        layoutSliders = QGridLayout()
        layoutSliders.addWidget(QLabel('Gauss-Blur - KernelSize'),0,0)
        layoutSliders.addWidget(self.label_gaussian_blur_kernel_size,0,1)
        layoutSliders.addWidget(self.slider1,0,2)
        
        layoutSliders.addWidget(QLabel('Canny - low threshold'),1,0)
        layoutSliders.addWidget(self.label_canny_low_threshold,1,1)
        layoutSliders.addWidget(self.slider2,1,2)
        
        layoutSliders.addWidget(QLabel('Canny - high threshold'),2,0)
        layoutSliders.addWidget(self.label_canny_high_threshold,2,1)
        layoutSliders.addWidget(self.slider3,2,2)
        
        layoutSliders.addWidget(QLabel('Hough - rho'),3,0)
        layoutSliders.addWidget(self.label_hough_rho,3,1)
        layoutSliders.addWidget(self.slider4,3,2)
        
        layoutSliders.addWidget(QLabel('Hough - theta'),4,0)
        layoutSliders.addWidget(self.label_hough_theta,4,1)
        layoutSliders.addWidget(self.slider5,4,2)
        
        layoutSliders.addWidget(QLabel('Hough - threshold'),5,0)
        layoutSliders.addWidget(self.label_hough_threshold,5,1)
        layoutSliders.addWidget(self.slider6,5,2)
        
        layoutSliders.addWidget(QLabel('Hough - min_line_lend'),6,0)
        layoutSliders.addWidget(self.label_hough_min_line_len,6,1)
        layoutSliders.addWidget(self.slider7,6,2)
        
        layoutSliders.addWidget(QLabel('Hough - max_line_gap'),7,0)
        layoutSliders.addWidget(self.label_hough_max_line_gap,7,1)
        layoutSliders.addWidget(self.slider8,7,2)
        
        layoutSliders.addWidget(self.pushButton_reset_parameters,8,0)
        
        return layoutSliders
    
   
        
    # ---------------------------
    def Create_FiguresLayout(self):
    
        # +--------------------------+
        # | Image1 | Image2 | Image3 |
        # +--------------------------+
   
        # ----------------------------------
        # Create the maptlotlib FigureCanvas objects, 
        self.sc1 = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc2 = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc3 = MplCanvas(self, width=5, height=4, dpi=100)
        
        # ----------------------------------
        # Create a grid layout with the 3 figures
        layout = QGridLayout()
        layout.addWidget(self.sc1,0,0)
        layout.addWidget(self.sc2,0,1)
        layout.addWidget(self.sc3,0,2)
        
        return layout

        
    # ---------------------------
    def Create_Toolbar_Statusbar(self):
        
        # Toolbar
        toolbar = QToolBar("Toolbar")
        toolbar.setIconSize(QSize(16,16))
        
        self.combobox_imagefiles = QComboBox()
        self.combobox_imagefiles.addItems(self._get_list_of_images())
        self.combobox_imagefiles.currentIndexChanged[str].connect(self._imagefile_changed)
        toolbar.addWidget(self.combobox_imagefiles)
                
        self.addToolBar(toolbar)
        
        # Statusbar
        self.setStatusBar(QStatusBar(self))
    
    def _get_list_of_images(self,folder = r'.\images'):
        ImageFiles = glob(os.path.join(folder,'*.bmp')) + glob(os.path.join(folder,'*.jpg'))
        ImageFiles.sort()
        return ImageFiles
        
    @pyqtSlot(str)
    def _imagefile_changed(self, s):
        self.update()
        
    def get_current_image_file_name(self):
       return str(self.combobox_imagefiles.currentText())
    
    # ------------------------------------------------
    def set_parameter_default_values(self):
        self.gaussian_blur_kernel_size=5
        self.canny_low_threshold=50
        self.canny_high_threshold=150
        
        # (x,y) 
        self.ROI_vertices_rel = [(0.10,1.00), (0.40,0.60), (0.55,0.60), (1.00,1.00)]
        
        self.hough_rho = 0.5                       # distance resolution in pixels of the Hough grid
        self.hough_theta = 0.5                     # angular resolution in radians of the Hough grid
        self.hough_threshold = 8                   # minimum number of votes (intersections in Hough grid cell)
        self.hough_min_line_len = 70               # minimum number of pixels making up a line
        self.hough_max_line_gap = 150              # maximum gap in pixels between connectable line segments
    
       
    def set_sliders(self):
        self.slider1.setValue(self.gaussian_blur_kernel_size)
        self.slider2.setValue(self.canny_low_threshold)
        self.slider3.setValue(self.canny_high_threshold)
        self.slider4.setValue(round(self.hough_rho*10.0))
        self.slider5.setValue(round(self.hough_theta*10.0))
        self.slider6.setValue(self.hough_threshold)
        self.slider7.setValue(self.hough_min_line_len)
        self.slider8.setValue(self.hough_max_line_gap)
       
    def display_current_parameter_values(self):
        self.label_gaussian_blur_kernel_size.setText('%d'%self.gaussian_blur_kernel_size)
        self.label_canny_low_threshold.setText('%d'%self.canny_low_threshold)
        self.label_canny_high_threshold.setText('%d'%self.canny_high_threshold)
        self.label_hough_rho.setText('%3.1f'%self.hough_rho)
        self.label_hough_theta.setText('%3.1f'%self.hough_theta)
        self.label_hough_threshold.setText('%d'%self.hough_threshold)
        self.label_hough_min_line_len.setText('%d'%self.hough_min_line_len)
        self.label_hough_max_line_gap.setText('%d'%self.hough_max_line_gap)
    
    
    # =========================
    # Slots
    # =========================
    @pyqtSlot(int)
    def update_gaussian_blur_kernel_size(self, value):
        gaussian_blur_kernel_size = value
        gaussian_blur_kernel_size += (gaussian_blur_kernel_size + 1) % 2
        gaussian_blur_kernel_size = max(1,gaussian_blur_kernel_size)
        
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.update()
    
    @pyqtSlot(int)
    def update_canny_low_threshold(self, value):
        self.canny_low_threshold = value
        self.update()

    @pyqtSlot(int)
    def update_canny_high_threshold(self, value):
        self.canny_high_threshold = value
        self.update()
    
    @pyqtSlot(int)
    def update_hough_rho(self, value):
        hough_rho = value/10.0;
        hough_rho = max(0.1,hough_rho)
        
        self.hough_rho = hough_rho
        self.update()

    @pyqtSlot(int)
    def update_hough_theta(self, value):
        hough_theta = value/10.0;
        hough_theta = max(0.1,hough_theta)
        
        self.hough_theta = hough_theta
        self.update()

    @pyqtSlot(int)
    def update_hough_threshold(self, value):
        self.hough_threshold = value
        self.update()

    @pyqtSlot(int)
    def update_hough_min_line_len(self, value):
        self.hough_min_line_len = value
        self.update()

    @pyqtSlot(int)
    def update_hough_max_line_gap(self, value):
        self.hough_max_line_gap = value
        self.update()

    @pyqtSlot()
    def on_click_reset_parametrs(self):
        print('Set tuning parameters to default values')
        self.set_parameter_default_values()
        self.set_sliders()
        self.display_current_parameter_values()
    
    # ==============================
    
    def update(self):
    
        #print('update')
        #print(self.gaussian_blur_kernel_size)
        #print(self.canny_low_self.hough_self.hough_threshold)
        #print(self.canny_high_self.hough_self.hough_threshold)
        #print(self.hough_rho)
        

        # ------------------
        # display current values
        self.display_current_parameter_values()
        

        # ------------------
        parameters = {}
        parameters['FileName'] = self.get_current_image_file_name()
        parameters['ROI_vertices_rel'] = self.ROI_vertices_rel
        
        parameters['gaussian_blur_kernel_size'] = self.gaussian_blur_kernel_size
        parameters['canny_low_threshold']       = self.canny_low_threshold 
        parameters['canny_high_threshold']      = self.canny_high_threshold
        parameters['hough_rho']                 = self.hough_rho
        parameters['hough_theta']               = self.hough_theta*np.pi/180.0
        parameters['hough_threshold']           = self.hough_threshold
        parameters['hough_min_line_len']        = self.hough_min_line_len
        parameters['hough_max_line_gap']        = self.hough_max_line_gap
        
        
        
        
        Images = self.iLaneFindingPipeline.process(parameters)
        
        self.sc1.axes.cla() 
        self.sc1.axes.imshow(Images['Original+ROI'])
        self.sc1.axes.set_title('Original Image + Region of Interres ROI')
        self.sc1.axes.set_xticks([])
        self.sc1.axes.set_yticks([])
        self.sc1.fig.tight_layout()
        self.sc1.draw()

        self.sc2.axes.cla() 
        self.sc2.axes.imshow(Images['edges'],cmap='gray')
        self.sc2.axes.set_title('Canny edges')
        self.sc2.axes.set_xticks([])
        self.sc2.axes.set_yticks([])
        self.sc2.fig.tight_layout()
        self.sc2.draw()
        
        self.sc3.axes.cla() 
        self.sc3.axes.imshow(Images['overlay_img'],cmap='gray')
        self.sc3.axes.set_title('Hough lines')
        self.sc3.axes.set_xticks([])
        self.sc3.axes.set_yticks([])
        self.sc3.fig.tight_layout()
        self.sc3.draw()
        
              
        self.show()
        #print('update-end')

# ===============================================
if __name__ == '__main__':
    app = QApplication(sys.argv)  

    window = MainWindow()
    window.show() # IMPORTANT!!!!! Windows are hidden by default.

    sys.exit(app.exec_())