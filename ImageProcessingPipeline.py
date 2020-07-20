# ImageProcessingPipeline
# 2020-07-20

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


class CHelper():
    
    def __init__(self, verbose=False, SwitchLineExtrapolate='default' ,SwitchDrawRawLines=False):
        self.verbose=verbose
        self.SwitchDrawRawLines = SwitchDrawRawLines
        self.SwitchLineExtrapolate = SwitchLineExtrapolate
        #self.SwitchLineExtrapolate='default'
        #self.SwitchLineExtrapolate='extrapolated'
        
        self._headings = []    # angles of the detected lines
        self._m_lefts = []     # slope left
        self._m_rights = []    # slope right
        self._indeces =[]      # time axis
        
        #debug
        self.ListLinesLeft = []
        self.ListLinesRight = []
        
        print('SwitchLineExtrapolate',self.SwitchLineExtrapolate)
        
    def GenerateNextImageIndex(self):
        self._indeces.append(len(self._indeces)+1)
    
    @property
    def currentIndex(self):
        return self._indeces[-1]

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  

        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
   

    @staticmethod
    def _CalcHeading(x1, y1, x2, y2):
        # heading = arctan(y/x)
        # heading in degree
        heading = math.degrees(math.atan2((y2-y1),(x2-x1)))
        return heading
    
    @staticmethod
    def _CheckIfInInterval(x, interval):
        x_min, x_max = interval
        if x_min>x_max:
            x_min,x_max = x_max,x_min
        return True if (x>x_min) and (x<x_max) else False

    
    def _furtherFilter(self, ListLines, mode,limit):
        headings = [ k[1] for k in ListLines]
        mean = np.mean(headings)
        if self.verbose:
            print('mean(heading):',mean)
        
        #std = np.std(headings)
        std = 5
        intervall = (mean-std,mean+std)
        
        '''
        ListLines2 = [ k[0] for k in ListLines]
        x = []
        y = []
        for line in ListLines2:
            for x1,y1,x2,y2 in line:
                x.append(x1)
                y.append(y1)
                x.append(x2)
                y.append(y2)
        if self.verbose:
            print('median:', np.median(x))
            print('mean:', np.mean(x))
            print('min:', np.min(x))
            print('max:', np.max(x))

        d1 = np.max(x)-np.median(x)
        d2 = np.median(x) - np.min(x)

        d = np.min([d1,d2])
        
        #mode = ''
        '''
        
        if self.verbose:
            print('mode:',mode)
            print('limit:',limit)
        ListLinesOutput = []
        for line, heading in ListLines:
            cond1 = CHelper._CheckIfInInterval(heading,intervall)
            for x1,y1,x2,y2 in line:
                if mode == 'right':
                    cond2 = (x1>limit) and (x2>limit)
                elif mode == 'right':
                    cond2 = (x1<limit) and (x2<limit)
                else:
                    cond2 = True
                if cond1 and cond2:
                    ListLinesOutput.append(line)

        return ListLinesOutput
    
    
    @staticmethod
    def _CalcLinearFit(ListLines, y_range):
        x = []
        y = []
        for line in ListLines:
            for x1,y1,x2,y2 in line:
                x.append(x1)
                y.append(y1)
                x.append(x2)
                y.append(y2)

        x = np.array(x).reshape((-1, 1))
        y = np.array(y)
        model = LinearRegression().fit(x, y)
        #print('intercept:', model.intercept_)
        #print('slope:', model.coef_)


        m = model.coef_
        b = model.intercept_

        y1,y2 = y_range
        # y=m*x+b -> x = (y-b)/m
        x1 = (y1-b)/m
        x2 = (y2-b)/m
        line = (int(x1),int(y1),int(x2),int(y2))
        return line, m, b
        
            
    def draw_lines_extrapolated(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to 
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).  

        Think about things like separating line segments by their 
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of 
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        
        ListLinesLeft = []
        ListLinesRight = []

        # Get image shape and define y region of interest value
        imshape = image.shape
        y_max   = imshape[0] # lines initial point at bottom of image    
        y_min   = imshape[0] 

        headings = []
        #heading_left_interval = (-15, -40)
        #heading_right_interval = (15, 40)
        heading_left_interval = (-25, -40)
        heading_right_interval = (25, 40)
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                y_min = np.min((y_min,y1,y2))

                heading  = self._CalcHeading(x1, y1, x2, y2)
                headings.append(heading)

                if self.verbose:
                    print('{} {}'.format((x1, y1, x2, y2),heading))

                if self._CheckIfInInterval(heading,heading_left_interval):
                    ListLinesLeft.append((line,heading))
                if self._CheckIfInInterval(heading,heading_right_interval):
                    ListLinesRight.append((line,heading))

        
        self._headings.extend(headings)
        
        

        #further filtering
        if True:
            if self.verbose:
                print('before',ListLinesLeft)
            ListLinesLeft = self._furtherFilter(ListLinesLeft,'left',int(imshape[1]*0.7))
            if self.verbose:
                print('after',ListLinesLeft)
            
            if self.verbose:
                print('before',ListLinesRight)
            ListLinesRight = self._furtherFilter(ListLinesRight,'right',int(imshape[1]*0.3))
            if self.verbose:
                print('after',ListLinesRight)
        else:
            ListLinesLeft = [ k[0] for k in ListLinesLeft]
            ListLinesRight = [ k[0] for k in ListLinesRight]
        
        # y_min
        y_min   = imshape[0] # init with maximal y value
        for line in ListLinesLeft:
            for x1,y1,x2,y2 in line:
                y_min = np.min((y_min,y1,y2))
        for line in ListLinesRight:
            for x1,y1,x2,y2 in line:
                y_min = np.min((y_min,y1,y2))
                
        #print('y_min',y_min)
        #y_min   = 280        # lines end point at top of ROI
        y_range = (y_min,y_max)
        
        self.ListLinesLeft = ListLinesLeft
        self.ListLinesRight = ListLinesRight
        
        
        
        AveragedLeftLine,  m_left,  b_left  =  self._CalcLinearFit(ListLinesLeft, y_range)
        AveragedRightLine, m_right, b_right =  self._CalcLinearFit(ListLinesRight, y_range)
        
        if self.verbose:
            print('AveragedLeftLine (%d)'%len(AveragedLeftLine))
            print(AveragedLeftLine)
            print('AveragedRightLine (%d)'%len(AveragedRightLine))
            print(AveragedRightLine)
        
        
        self._m_lefts.append(m_left)
        self._m_rights.append(m_right)
            
        # define average left and right lines
        self.draw_lines(img, [[AveragedLeftLine],[AveragedRightLine]], color=color, thickness=thickness)
       
        # selected raw lines
        if self.SwitchDrawRawLines:
            self.draw_lines(img, ListLinesLeft+ListLinesRight, color=[0, 255, 0], thickness=thickness)
        
    @property
    def headings(self):
        return np.array(self._headings)
    
    @property
    def m_lefts(self):
        return np.array(self._m_lefts)
    @property
    def m_rights(self):
        return np.array(self._m_rights)
    
    @property
    def indeces(self):
        return np.array(self._indeces)
    
    # --------------------------------------
    @staticmethod
    def grayscale(img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def gaussian_blur(img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @staticmethod
    def region_of_interest(img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    @staticmethod
    def canny(img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)


    @staticmethod
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns hough lines 
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
      
        return  lines
    

    # -----------------------------------------------------------------
    # Python 3 has support for cool math symbols.
    @staticmethod
    def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + γ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, γ)
#==============================================================
def get_ROI_vertices(ROI_vertices_rel, imshape):
 
    ROI_vertices =[]
    
    for x,y in ROI_vertices_rel:
        ROI_vertices.append((imshape[1]*x, imshape[0]*y))
        
    ROI_vertices = np.array([ROI_vertices], dtype=np.int32)
    return ROI_vertices
        


        
#==============================================================
class CLaneFindingPipeline(CHelper):
    
    def __init__(self, parameterset = 1, SwitchLineExtrapolate='default', verbose = False, SwitchSaveImages=False,OutputFolderName="",SwitchDrawRawLines=False):
        super().__init__( verbose = verbose, SwitchLineExtrapolate=SwitchLineExtrapolate, SwitchDrawRawLines=SwitchDrawRawLines)
        
        self.parameterset = parameterset
   
        
    def process(self, parameters):
        
        Images = {}
        
        # parameters-"package" -> local variables 
        FileName         = parameters['FileName']
        ROI_vertices_rel = parameters['ROI_vertices_rel']
        
        gaussian_blur_kernel_size = parameters['gaussian_blur_kernel_size']
        canny_low_threshold       = parameters['canny_low_threshold']
        canny_high_threshold      = parameters['canny_high_threshold']
        
        hough_rho                 = parameters['hough_rho']
        hough_theta               = parameters['hough_theta']
        hough_threshold           = parameters['hough_threshold']
        hough_min_line_len        = parameters['hough_min_line_len']
        hough_max_line_gap        = parameters['hough_max_line_gap']
        
        
        # ----------------------------------
        # Load image and display 
        image = mpimg.imread(FileName)  
        
        image2 = image.copy()
        # https://www.geeksforgeeks.org/python-opencv-cv2-polylines-method/
        color = (255, 0, 0) 
        thickness = 2
        isClosed = True
        ROI_vertices = get_ROI_vertices(ROI_vertices_rel, image2.shape)
        image2 = cv2.polylines(image2, [ROI_vertices], isClosed, color, thickness) 
        
        Images['Original+ROI'] = image2
        
        
        # --------------------------------
        # image pipeline
        imshape = image.shape
        gray = self.grayscale(image)
        
        # Define a kernel size and apply Gaussian smoothing
        blur_gray = self.gaussian_blur(gray, kernel_size=gaussian_blur_kernel_size)

        # edge detection
        edges = self.canny(blur_gray, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold)

        Images['edges'] = edges
        
        
        # -------------------------------
        # Mask - Region of Interest ROI 
        masked_image = self.region_of_interest(edges, ROI_vertices)


        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        lines = self.hough_lines(masked_image, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap)


        # draw hough lines into a new image of same size
        line_img = np.zeros((masked_image.shape[0], masked_image.shape[1], 3), dtype=np.uint8)
        if self.SwitchLineExtrapolate=='extrapolate':
            self.draw_lines_extrapolated(line_img, lines)
        else:
            self.draw_lines(line_img, lines)


        # combine detected lines and image
        overlay_img = self.weighted_img(line_img, image, α=0.8, β=1., γ=0.)

        Images['overlay_img'] = overlay_img
        
        
        # -------------------------------

        return Images
    
    