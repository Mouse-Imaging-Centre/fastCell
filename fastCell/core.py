import cv2 as cv
import numpy as np
from typing import List, Union

def random_cells(cluster: np.ndarray, cell_area: float) -> List:

    num_cells = cv.contourArea(cluster)//cell_area
    n = 0
    centroids = []
    x, y, w, h = cv.boundingRect(cluster)
    while n < num_cells:
        center = np.random.randint(low=x, high=x + w), np.random.randint(low=y, high=y + h)
        if cv.pointPolygonTest(contour=cluster, pt=center, measureDist=False) == 1:
            centroids.append(center)
            n+=1
    return centroids