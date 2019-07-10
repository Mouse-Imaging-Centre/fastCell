import argparse
import cv2 as cv
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Segment the cells from an image.')
parser.add_argument(dest="segment", type=str,
                    help = "Segmentation to pixelize")
parser.add_argument(dest="centroids", type=str,
                    help="Write out each cell as pixel.")
parser.add_argument("--centroid-intensity", dest="centroid_intensity", type=int, default=255)

args = parser.parse_args()

if __name__ == '__main__':

    segment = cv.imread(args.segment, cv.COLOR_BGR2GRAY)

    contours, hierarchy = cv.findContours(segment.astype("uint8"), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # cv.findContours returns a list of np.ndarray of shape [px, unknown, 2].
    contours = [np.squeeze(contour, axis=1) for contour in contours]
    df = pd.DataFrame({'contour': contours}).assign(
        moments=lambda df: df.contour.apply(lambda contour: cv.moments(contour)),
        area=lambda df: df.contour.apply(lambda contour: cv.contourArea(contour)),
        perimeter=lambda df: df.contour.apply(lambda contour: cv.arcLength(contour, closed=True))
    )
    df = df.assign(
        centroid=lambda df: df.moments.apply(lambda moments:
                                             (int(moments['m10'] / moments['m00']),
                                              int(moments['m01'] / moments['m00']))
                                             )
    )

    centroids = np.zeros(segment.shape, np.uint8)
    for centroid in df.centroid:
        cv.circle(img=centroids,
                  center=centroid,
                  radius=0,
                  color=args.centroid_intensity,
                  thickness=1)
    cv.imwrite(args.centroids, centroids)