# !pip install opencv-python #openCV may need to be installed before running code
import cv2
import numpy as np
import glob
# %%
class ColourQR:
    """
    Instructions:
        - Ensure openCV library is installed before running ColourQR class
        - Add folder path as parameter for ColourQR and call the final function ‘colourMatrix()’
        - ColourQR will import all images from the folder path and apply image processing functions
        - ColourQR will return the original image, corrected image and the colour matrix for the corrected image
    """

    def __init__(self, folderpath='folderpath'):
        self.folderpath = folderpath

    def loadImage(self):
        """
        :return: List of images imported in alphabetical order of filename as NP arrays of size 512x512 in RGB colour space
        """
        filenames = glob.glob(self.folderpath + '/*.*')  # Will import all file types in folder path
        filenames.sort()  # Import images in alphabetical order
        images = []
        for file in filenames:
            img = cv2.imread(file)  # Import image as BGR colour space by CV2
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB colour space
            img_RGB = cv2.resize(img_RGB, (512, 512),
                                 interpolation=cv2.INTER_AREA)  # Resize to 512x512, needed for NumPy array of images
            images.append(img_RGB)

        images = np.array(images)
        return images

    def imagePreProcessing(self, img):
        """
        :param img: Input image should be in RGB colour space
        :return: Blurred image with noise removed in RGB colour space
        """
        # Blur and denoise img in RGB colour space
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)  # 5x5 kernel as lots of noise in the images
        img_denoise = cv2.fastNlMeansDenoisingColored(img_blur, 20, 10, 7, 21)  # Denoise using NLMeans filter
        return img_denoise

    def removeColour(self, img):
        """
        :param img: Input image should be in RGB colour space and have been de-noised
        :return: Image with areas of colour filled with white, leaving black circles, in RGB colour space
        """
        # Define HSV thresholds in order to create a mask of all colours in image excluding black and white pixels
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 20, 0])
        upper_hsv = np.array([180, 255, 255])
        mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        img_masked = cv2.bitwise_and(img, img, mask=mask)

        # Dilate masked image to accentuate edges and perform canny edge detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Large kernel with ellipse structuring element
        img_dil = cv2.dilate(img_masked, kernel, iterations=2)
        img_canny = cv2.Canny(img_dil, 150, 250)  # Tight canny edge detection

        # Identify contours of image
        contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)  # Just want to find external edges
        colour_filled = img.copy()
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Sort list of contours by largest area
        colour_filled = cv2.fillPoly(colour_filled, pts=[largest_contours[0]],
                                     color=(255, 255, 255))  # Fill the largest area with white
        return colour_filled

    def findCircles(self, img):
        """
        :param img: Image should be in RGB colour space and have had coloured pixels replaced with white pixels
        :return: List of black circle centroid coordinates (x4)
        """
        # Convert filled image to grayscale and threshold for only black circles
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_thresh = cv2.threshold(img_gray, 60, 255,
                                        cv2.THRESH_BINARY_INV)  # Strict inverse binary threshold to identify only black pixels in the circles

        # Dilate image to accentuate black circles and to remove any white pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (3, 3))  # Smaller kernel compared to before and fewer iterations
        img_dil = cv2.dilate(img_thresh, kernel, iterations=1)

        # Find bounding contours of the four black circles
        contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[
                           :4]  # Sort list of contours by area and keep four largest contour areas

        # Find the 'moment' of each contour which allows to locate the centroid of the circle
        circle_cntr = []
        for c in largest_contours:  # Compute the centroid of each contour, these should be of the four black circles
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            circle_cntr.append((cX, cY))  # Append x and y coordinates of each centroid to the list

        # Sort the list of centroid coordinates by the difference of the coordinates from smallest to largest then order
        circle_cntr = sorted(circle_cntr, key=lambda sub: abs(sub[0] - sub[1]), reverse=False)
        circle2 = circle_cntr[:2]
        circle3 = circle_cntr[2:]
        circle2 = sorted(circle2, reverse=False)
        circle3 = sorted(circle3, reverse=False)
        circle_cntr = circle2 + circle3
        return circle_cntr

    def correctImage(self, img, circle_cntr):
        """
        :param img: Image should be in RGB colour space and have been de-noised
        :param circle_cntr: List of four circle centroid coordinates
        :return: Corrected image in RGB colour space, where the original warped image is projected using the co-ordinates of the four centroids
        """
        correct_circles = np.array([(27, 27), (474, 474), (27, 474), (
        474, 27)])  # Define the ground truth circle centroid coordinates for image projection
        projection = np.array(circle_cntr)  # Convert centroid coordinates to a np array
        h, mask = cv2.findHomography(projection,
                                     correct_circles)  # Find the homography matrix based on the provided centroid coordinates
        img_corrected = cv2.warpPerspective(img, h, (
        512, 512))  # Using warPerspective, project warped image using ground truth centroid coordinates
        return img_corrected

    def findSquares(self, img):
        """
        :param img: Image should be in RGB colour space and should have been corrected for any distortion
        :return: List of the centroid co-ordinates for each of the 16 squares in the 4x4 QR code
        """
        # Define HSV thresholds in order to create a mask of all colours in the corrected image excluding black and white pixels
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 20, 0])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(img_hsv, lower, upper)
        img_masked = cv2.bitwise_and(img, img, mask=mask)

        # Dilate masked image to accentuate edges and perform canny edge detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Large kernel with ellipse morphology
        img_dil = cv2.dilate(img_masked, kernel,
                             iterations=4)  # More iterations than before in order to blend colours across white space inside main square
        img_canny = cv2.Canny(img_dil, 150, 250)

        # Get bounding contours of the area of colour in the image
        contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contours = np.array(
            largest_contours[0])  # Contour list sorted by greatest area and highest value extracted
        x, y, w, h = cv2.boundingRect(
            largest_contours)  # Find x,y starting coordinates, height and width of bounding rectangle of colour area of image
        x = x + 10  # Adjust x, y, w and h to reduce rectangle, accounting for large dilation previously applied
        y = y + 10
        w = w - 20
        h = h - 20

        # Each image has the same number of squares, 4x4. Use this to segment the bounding rectangle into a 4x4 matrix
        gridHeight = float(
            h / 4)  # Estimate height and width of each small square by dividing bounding rectangle h and w by 4
        gridWidth = float(w / 4)

        # Define each square and find the centroid which is later used to sample colour
        square_centres = []
        for i in range(0, 4, 1):
            small_y = int(i * gridHeight)
            for j in range(0, 4, 1):
                small_x = int(j * gridWidth)
                small_x2 = x + small_x
                small_y2 = y + small_y
                small_w2 = int(w / 4)
                small_h2 = int(h / 4)
                centre_x = int(small_x2 + (0.5 * small_w2))  # Find center coordinates of each square
                centre_y = int(small_y2 + (0.5 * small_h2))
                square_centres.append((centre_x,
                                       centre_y))  # List of small square centroid coordinates as tuples for pixel sampling of colours
        return square_centres

    def findColours(self, img, square_centres):
        """
        :param img: Image should be in RGB colour space and should have been corrected for any distortion
        :param square_centres: List of the centroid co-ordinates for each of the 16 squares in the 4x4 QR code
        :return: A numpy 4x4 matrix of image colours represented as W, R, Y, G and B, based on corrected input image
        """
        # Convert original RGB img to HSV colour space for colour detection
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        list_colours_HSV = []

        # Iterate through list of small square centroid coordinates and extract the HSV value fo the pixel
        for index, t in enumerate(square_centres):
            x = t[0]
            y = t[1]
            h, s, v = img_hsv[
                x, y]  # CV2 uses different elementwise values to define HSV, can then correct the values for h, s and v for easier colour thresholding
            h = h * 2
            s = s / 255
            v = v / 255
            a = (h, s, v)
            list_colours_HSV.append(a)  # Append corrected HSV value as 3-tuple

        colours = []  # Colour identification by thresholding HSV value for each square centroid
        for index, t in enumerate(list_colours_HSV):
            h = t[0]
            s = t[1]
            if s < 0.1:  # Threshold on saturation for white and append 'W'
                colours.append('w')
            elif h <= 25:  # Threshold on hue for red and append 'R'
                colours.append('r')
            elif h <= 75:  # Threshold on hue for yellow and append 'Y'
                colours.append('y')
            elif h <= 180:  # Threshold on hue for green and append 'G'
                colours.append('g')
            elif h <= 255:  # Threshold on hue for blue and append 'B'
                colours.append('b')

        colour_matrix = [[colours[0], colours[4], colours[8], colours[12]],
                         # Define new list to create colour matrix of identified colours
                         [colours[1], colours[5], colours[9], colours[13]],
                         [colours[2], colours[6], colours[10], colours[14]],
                         [colours[3], colours[7], colours[11], colours[15]]]
        colour_matrix = np.array(colour_matrix)  # Convert matrix to a np array
        return colour_matrix

    def colourMatrix(self):
        """
        :return: List of numpy 4x4 colour matrices for all images in a given folder path
        """
        colour_matrices = []
        images = self.loadImage()
        for image in images:
            img = image
            img_RGB = self.imagePreProcessing(img)  # Returns RGB img with noise removed
            img_filled = self.removeColour(img_RGB)  # Returns RGB img with colours filled in white
            circle_centres = self.findCircles(img_filled)  # Returns list of circle centroids
            img_corrected = self.correctImage(img_RGB, circle_centres)  # Transforms image and returns corrected RGB img
            square_centres = self.findSquares(img_corrected)  # Returns list of small square centroids
            colours = self.findColours(img_corrected, square_centres)  # Returns colour matrix of image
            colour_matrices.append(colours)
            for line in colours:  # Print matrix in correct format
                print('  '.join(map(str, line)))
        return colour_matrices


# %%
# Input folder path here:
ColourQR(
    folderpath='/Users/robbie/Documents/!Uni-Data_Science_Masters/S2-Image_Processing/4-Assignment/images2 2/Noise').colourMatrix()