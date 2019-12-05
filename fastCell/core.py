import cv2 as cv
import numpy as np
from typing import List, Union, Callable
import argparse

def parse_and_validate(parser: argparse.ArgumentParser, validators: List[Callable] = []) -> argparse.Namespace:
    args = parser.parse_args()
    for validate in validators:
        validate(parser, args)
    return args

def add_default_groups(master_parser: argparse.ArgumentParser) -> None:
    for add_group in [add_application_group, add_input_group, add_output_group]:
        add_group(master_parser)

def add_application_group(master_parser: argparse.ArgumentParser) -> None:
    application_group = master_parser.add_argument_group("Application")
    application_group.add_argument("--temp-dir", dest="temp_dir", type=str, default=None,
                                   help="Please do not use any parent directory notation like .. on UNIX systems.")
    application_group.add_argument("--keep-temp", dest="keep_temp", action="store_true", default=False)
    application_group.add_argument("--verbose", dest="verbose", action="store_true", default=False)

def add_input_group(master_parser: argparse.ArgumentParser) -> None:
    input_group = master_parser.add_argument_group("Input")
    input_group.add_argument("--image", dest="image", type=str, required=True,
                             help="Input Image")

def add_output_group(master_parser: argparse.ArgumentParser) -> None:
    output_group = master_parser.add_argument_group("Output")
    output_group.add_argument("--segment-output", dest="segment_output", type=str, required=False,
                              help="Write out the segmentation.")
    output_group.add_argument("--segment-intensity", dest="segment_intensity", type=int, default=255,
                              help="default: %(default)s")
    output_group.add_argument("--centroids-output", dest="centroids_output", type=str, required=False,
                              help="Write out each cell as pixel.")
    output_group.add_argument("--centroid-intensity", dest="centroid_intensity", type=int, default=1,
                              help="default: %(default)s")
    output_group.add_argument("--outlines-output", dest="outlines_output", type=str, required=False,
                              help="Outline the identified cells.")
    output_group.add_argument("--image-output", dest="image_output", type=str, required=False,
                              help="Write out the image. It may have been cropped or otherwise processed.")


def validate_input_args(args) -> None:
    pass
def validate_output_args(args) -> None:
    pass

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