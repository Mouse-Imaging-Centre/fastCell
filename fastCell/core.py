import cv2 as cv
import numpy as np

class Cell:
    '''
    Create a Cell instance as follows: my_cell = Cell(image, contour)
        where image is the image and contour is returned from cv.findContours()

    Attributes:

    '''

    def __init__(self, image, contour):
        self.image = image,
        self.contour = contour,

        self.mask = np.zeros(image.shape, np.uint8)
        cv.drawContours(self.mask, [contour], idx = 0, color = 255, thickness = -1)
        # this would return only the indices of the cell
        # self.mask = np.transpose(np.nonzero(self.mask))
        # self.mask = cv.findNonZero(self.mask)

        self.min_val, self.max_val, self.min_loc, self.max_loc = cv.minMaxLoc(image, mask = self.mask)
        self.mean_val = cv.mean(image, mask=self.mask)

        self.moments = cv.moments(contour)
        self.centroid = (int(self.moments['m10'] / self.moments['m00']),
                         int(self.moments['m01'] / self.moments['m00']))

        self.area = cv.contourArea(contour)
        self.perimeter = cv.arcLength(contour, True)
        self.equivalent_diameter = np.sqrt(4 * self.area / np.pi)

        self.convex_hull = cv.convexHull(contour)
        self.convex_area = cv.contourArea(self.convex_hull)
        self.solidity = self.area / float(self.convex_area)

        self.ellipse = cv.fitEllipse(contour)
        (self.center, self.axes, self.orientation) = self.ellipse
        self.majoraxis_length = max(self.axes)
        self.minoraxis_length = min(self.axes)
        self.eccentricity = np.sqrt(1 - (self.minoraxis_length / self.majoraxis_length) ** 2)
